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
        token_param = os.getenv("GW_TOKENS_PARAM", "max_tokens").strip() or "max_tokens"
        kwargs[token_param] = int(max_tokens)
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)
            if "Unsupported parameter" in msg and "max_tokens" in msg:
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = int(max_tokens)
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            raise

    return _with_backoff(_do_call)
