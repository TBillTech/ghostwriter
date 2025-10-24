"""LLM client abstraction and completion helpers.

This is a minimal extraction from the current driver to enable modular use.
Future refactors will migrate scripts/driver.py to use this module.
"""
from __future__ import annotations

import os
import re
import random
import time
from typing import Optional, List
from .logging import breadcrumb as _breadcrumb, log_run as _log_run
from .tokenizer import count_chat_tokens as _count_chat_tokens

try:
    from openai import OpenAI
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception:  # pragma: no cover
        AzureOpenAI = None  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

_CLIENT = None  # type: ignore
_CLIENT_INFO = ""


def get_model() -> str:
    return os.getenv("GW_MODEL_DEFAULT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _with_backoff(fn, *, retries=3, base_delay=1.0, jitter=0.2):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(base_delay * (2 ** i) + random.random() * jitter)
    if last_err:
        raise last_err


def get_client():
    global _CLIENT, _CLIENT_INFO
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint and AzureOpenAI is not None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            ak = os.getenv("AZURE_OPENAI_API_KEY") or api_key
            _CLIENT = AzureOpenAI(azure_endpoint=azure_endpoint, api_version=api_version, api_key=ak)
            _CLIENT_INFO = f"azure:{azure_endpoint}|v={api_version}"
            return _CLIENT

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if base_url:
            bu = base_url.strip()
            lower = bu.lower()
            is_azure = ("azure.com" in lower) or ("openai.azure" in lower)
            if (not is_azure) and not re.search(r"/v\d+/?$", bu):
                bu = bu.rstrip("/") + "/v1"
            _CLIENT = OpenAI(base_url=bu, api_key=api_key)
            _CLIENT_INFO = f"base_url:{bu}"
        else:
            _CLIENT = OpenAI(api_key=api_key)
            _CLIENT_INFO = "default"
        return _CLIENT
    except Exception:
        return None


def complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Chat-completion with graceful mock fallback if no API key/client.
    """
    client = get_client()
    if client is None:
        head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
        return f"[MOCK LLM RESPONSE]\nSystem: {system or 'n/a'}\nTemp: {temperature}\n---\n{head}"

    model_name = model or get_model()

    def _do_call():
        messages = [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if reasoning_effort and str(reasoning_effort).strip().lower() in ("low", "medium", "high"):
            kwargs["reasoning_effort"] = str(reasoning_effort).strip().lower()
            _breadcrumb(f"llm2:reasoning_effort={kwargs['reasoning_effort']}")
        token_param = os.getenv("GW_TOKENS_PARAM", "max_tokens").strip() or "max_tokens"
        kwargs[token_param] = int(max_tokens)
        # Preflight prompt token accounting and request summary
        try:
            ptoks = int(_count_chat_tokens(messages, model_name))
        except Exception:
            ptoks = 0
        try:
            chars = sum(len(str(m.get("content", "")) or "") for m in messages)
        except Exception:
            chars = 0
        cpt = os.getenv("GW_CHARS_PER_TOKEN", "4") or "4"
        try:
            _breadcrumb(f"llm2:prompt_tokens tiktoken={ptoks} chars={chars} cpt={float(cpt) if cpt else 4.0}")
            _log_run(
                f"LLM request | model={model_name} temp={temperature} param={token_param} "
                f"prompt_tokens={ptoks} chars={chars} cpt={cpt} limit={kwargs[token_param]} "
                f"reasoning_effort={kwargs.get('reasoning_effort','n/a')}"
            )
        except Exception:
            pass
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)
            if "Unsupported parameter" in msg and "max_tokens" in msg:
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = int(max_tokens)
                resp = client.chat.completions.create(**kwargs)
                out = resp.choices[0].message.content or ""
                try:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        pt = getattr(usage, "prompt_tokens", None)
                        ct = getattr(usage, "completion_tokens", None)
                        rt = getattr(usage, "reasoning_tokens", None)
                        tt = getattr(usage, "total_tokens", None)
                        _log_run(f"LLM response | usage prompt={pt} completion={ct} reasoning={rt} total={tt}")
                except Exception:
                    pass
                return out
            raise
    return _with_backoff(_do_call)
