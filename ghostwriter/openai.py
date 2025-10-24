"""OpenAI client utilities and completion wrapper.

This module centralizes LLM client creation and a thin completion wrapper
with robust fallbacks and diagnostics. It depends only on environment variables
and the ghostwriter.logging helpers for breadcrumbs.

Swap-friendly: In the future, you can provide an alternative module (e.g.,
`ghostwriter/gemini.py`) that exposes the same public API and update imports
accordingly.

Public API:
- get_client() -> client | None
- get_model() -> str
- with_backoff(fn, retries=3, base_delay=1.0, jitter=0.2)
- llm_complete(prompt: str, system: Optional[str], temperature: float, max_tokens: int, model: Optional[str]) -> str
"""
from __future__ import annotations
from typing import Optional, List
import os
import re
import time
import random
import hashlib
import math

from .logging import breadcrumb as _breadcrumb, log_run as _log_run
from .tokenizer import count_chat_tokens as _count_chat_tokens

try:
    # OpenAI v2 client (requirements pinned to openai==2.x)
    from openai import OpenAI
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception:  # pragma: no cover
        AzureOpenAI = None  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

_CLIENT = None  # type: ignore
_CLIENT_INFO = ""  # for diagnostics (base_url or azure endpoint)


def get_client():
    """Return OpenAI client if API key is present; otherwise None for mock mode.
    Supports native OpenAI and Azure OpenAI. Does not attempt OpenRouter.
    """
    global _CLIENT, _CLIENT_INFO
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        # Azure OpenAI support
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        if azure_endpoint and AzureOpenAI is not None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            ak = os.getenv("AZURE_OPENAI_API_KEY") or api_key
            client = AzureOpenAI(azure_endpoint=azure_endpoint, api_version=api_version, api_key=ak)
            _CLIENT = client
            _CLIENT_INFO = f"azure:{azure_endpoint}|v={api_version}"
            _breadcrumb(f"openai:client-initialized azure_endpoint={azure_endpoint} api_version={api_version}")
            return _CLIENT

        # Normalize base URL if provided; many providers require '/v1' suffix
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        client = None
        if base_url:
            bu = base_url.strip()
            # Heuristic: if not azure-like and missing '/v1' suffix, append it
            lower = bu.lower()
            is_azure = ("azure.com" in lower) or ("openai.azure" in lower)
            if (not is_azure) and not re.search(r"/v\d+/?$", bu):
                bu = bu.rstrip("/") + "/v1"
            client = OpenAI(base_url=bu, api_key=api_key)
            _CLIENT = client
            _CLIENT_INFO = f"base_url:{bu}"
            _breadcrumb(f"openai:client-initialized base_url={bu}")
        else:
            client = OpenAI(api_key=api_key)
            _CLIENT = client
            _CLIENT_INFO = "default"
            _breadcrumb("openai:client-initialized base_url=default")
        return _CLIENT
    except Exception:
        return None


def get_model() -> str:
    # Prefer GW_MODEL_DEFAULT; fall back to legacy OPENAI_MODEL for compatibility
    return os.getenv("GW_MODEL_DEFAULT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def with_backoff(fn, *, retries: int = 3, base_delay: float = 1.0, jitter: float = 0.2):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(base_delay * (2 ** i) + random.random() * jitter)
    if last_err:
        raise last_err


def llm_complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: Optional[str] = None,
) -> str:
    """Call OpenAI if configured; else return a deterministic mock response."""
    _breadcrumb("llm:enter")
    client = get_client()
    if client is None:
        # Mock response for offline/dev usage
        head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
        tail = ("..." + prompt[-220:]) if len(prompt) > 220 else prompt
        h = hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:12]
        out = (
            f"[MOCK LLM RESPONSE]\nSystem: {system or 'n/a'}\nTemp: {temperature}\nHash:{h}\n"
            f"---\nHEAD:\n{head}\n---\nTAIL:\n{tail}"
        )
        _breadcrumb("llm:mock-return")
        return out

    model_name = model or get_model()

    def _estimate_prompt_tokens(texts: List[str]) -> int:
        """Roughly estimate tokens from text length; configurable chars-per-token.
        Uses GW_CHARS_PER_TOKEN (default 4). Adds optional GW_PROMPT_TOKENS_PAD tokens.
        """
        try:
            cpt = float(os.getenv("GW_CHARS_PER_TOKEN", "4").strip() or "4")
            if cpt <= 0:
                cpt = 4.0
        except Exception:
            cpt = 4.0
        total_chars = 0
        for t in texts:
            try:
                total_chars += len(t or "")
            except Exception:
                continue
        est = int(math.ceil(total_chars / cpt))
        try:
            pad = int(os.getenv("GW_PROMPT_TOKENS_PAD", "0") or "0")
            if pad > 0:
                est += pad
        except Exception:
            pass
        return max(est, 0)

    def _do_call():
        # Prefer chat.completions for richer prompting
        _breadcrumb(f"llm:chat.create:before model={model_name} endpoint={_CLIENT_INFO}")
        # Allow env to force token param key (max_tokens vs max_completion_tokens)
        token_param = os.getenv("GW_TOKENS_PARAM", "max_tokens").strip() or "max_tokens"
        _breadcrumb(f"llm:tokens-param={token_param}")
        # Reasoning effort (reasoning models only: 'low'|'medium'|'high')
        reasoning_effort = (os.getenv("GW_REASONING_EFFORT") or os.getenv("GW_REASONING_EFFORT_DEFAULT") or "").strip().lower()
        messages = [
            {"role": "system", "content": system or "You are a helpful writing assistant."},
            {"role": "user", "content": prompt},
        ]
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if reasoning_effort in ("low", "medium", "high"):
            kwargs["reasoning_effort"] = reasoning_effort
            _breadcrumb(f"llm:reasoning_effort={reasoning_effort}")
        # Optionally include prompt tokens estimate in the limit if provider expects total tokens
        try:
            include_prompt = os.getenv("GW_INCLUDE_PROMPT_TOKENS", "0").strip() == "1"
        except Exception:
            include_prompt = False
        # Real token counting (when tiktoken available)
        prompt_count = 0
        prompt_chars = 0
        try:
            prompt_count = int(_count_chat_tokens(messages, model_name))
            # Char-based estimate
            try:
                prompt_chars = sum(len(str(m.get("content", "")) or "") for m in messages)
            except Exception:
                prompt_chars = 0
            try:
                cpt = float(os.getenv("GW_CHARS_PER_TOKEN", "4") or "4")
                if cpt <= 0:
                    cpt = 4.0
            except Exception:
                cpt = 4.0
            char_est = int((prompt_chars / cpt) + 0.5)
            _breadcrumb(f"llm:prompt_tokens tiktoken={prompt_count} chars={prompt_chars} cpt={cpt} char_est={char_est}")
        except Exception:
            prompt_count = 0
            prompt_chars = 0
        effective_limit = int(max_tokens)
        # For max_completion_tokens, do NOT include prompt tokens; cap and buffer if configured
        if token_param == "max_completion_tokens":
            if include_prompt:
                _breadcrumb("llm:include-prompt-tokens:ignored-for-max_completion_tokens")
            try:
                comp_cap = int(os.getenv("GW_MAX_COMPLETION_TOKENS_CAP", "128000") or "128000")
            except Exception:
                comp_cap = 128000
            try:
                comp_buffer = int(os.getenv("GW_COMPLETION_BUFFER", "0") or "0")
            except Exception:
                comp_buffer = 0
            try:
                comp_min = int(os.getenv("GW_MIN_COMPLETION_TOKENS", "100") or "100")
            except Exception:
                comp_min = 100
            # Cap, then subtract buffer, then floor
            effective_limit = min(effective_limit, comp_cap)
            if comp_buffer > 0:
                effective_limit = max(comp_min, effective_limit - comp_buffer)
            _breadcrumb(f"llm:completion_cap cap={comp_cap} buffer={comp_buffer} min={comp_min} eff={effective_limit}")
        elif include_prompt:
            prompt_est = prompt_count or _estimate_prompt_tokens([
                str(messages[0]["content"] or ""),
                str(messages[1]["content"] or ""),
            ])
            # Avoid overflow if caller already specified a total-like value
            # We cap at a reasonable upper bound if user sets very high defaults
            try:
                cap = int(os.getenv("GW_MAX_TOKENS_CAP", "64000") or "64000")
            except Exception:
                cap = 64000
            effective_limit = min(max_tokens + prompt_est, cap)
            _breadcrumb(f"llm:include-prompt-tokens est={prompt_est} eff_limit={effective_limit}")
        kwargs[token_param] = effective_limit
        # Emit a one-line summary before the call
        try:
            _log_run(
                f"LLM request | model={model_name} temp={temperature} param={token_param} "
                f"prompt_tokens={prompt_count} chars={prompt_chars} cpt={os.getenv('GW_CHARS_PER_TOKEN','4')} "
                f"limit={effective_limit} reasoning_effort={reasoning_effort or 'n/a'}"
            )
        except Exception:
            pass
        try:
            resp = client.chat.completions.create(**kwargs)
            _breadcrumb("llm:chat.create:after")
            out = resp.choices[0].message.content or ""
            # Log usage summary if provided by the API
            try:
                usage = getattr(resp, "usage", None)
                if usage:
                    pt = getattr(usage, "prompt_tokens", None)
                    ct = getattr(usage, "completion_tokens", None)
                    rt = getattr(usage, "reasoning_tokens", None)
                    tt = getattr(usage, "total_tokens", None)
                    _breadcrumb(f"llm:usage prompt={pt} completion={ct} reasoning={rt} total={tt}")
                    try:
                        fr = None
                        try:
                            fr = (resp.choices[0].finish_reason if resp.choices and resp.choices[0] else None)
                        except Exception:
                            fr = None
                        _log_run(f"LLM response | usage prompt={pt} completion={ct} reasoning={rt} total={tt} finish_reason={fr}")
                    except Exception:
                        pass
            except Exception:
                pass
            # If the model returned empty content, retry once with a higher token budget
            if not out.strip():
                try:
                    inc = int(os.getenv("GW_RETRY_TOKEN_INCREMENT", "0") or "0")
                except Exception:
                    inc = 0
                if inc > 0:
                    try:
                        # Increase the effective limit on the selected token param
                        new_limit = effective_limit + inc
                        try:
                            cap = int(os.getenv("GW_MAX_TOKENS_CAP", "64000") or "64000")
                        except Exception:
                            cap = 64000
                        new_limit = min(new_limit, cap)
                        kwargs[token_param] = new_limit
                        _breadcrumb(f"llm:empty-retry:inc_tokens by={inc} new_limit={new_limit}")
                        resp = client.chat.completions.create(**kwargs)
                        _breadcrumb("llm:chat.create:after")
                        out2 = resp.choices[0].message.content or ""
                        try:
                            usage2 = getattr(resp, "usage", None)
                            if usage2:
                                pt2 = getattr(usage2, "prompt_tokens", None)
                                ct2 = getattr(usage2, "completion_tokens", None)
                                rt2 = getattr(usage2, "reasoning_tokens", None)
                                tt2 = getattr(usage2, "total_tokens", None)
                                _breadcrumb(f"llm:usage prompt={pt2} completion={ct2} reasoning={rt2} total={tt2}")
                                try:
                                    fr2 = None
                                    try:
                                        fr2 = (resp.choices[0].finish_reason if resp.choices and resp.choices[0] else None)
                                    except Exception:
                                        fr2 = None
                                    _log_run(f"LLM response | usage prompt={pt2} completion={ct2} reasoning={rt2} total={tt2} finish_reason={fr2}")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        if out2.strip():
                            _breadcrumb("llm:exit")
                            return out2
                    except Exception:
                        pass
            _breadcrumb("llm:exit")
            return out
        except Exception as e:
            # Optional fallback on 404-like HTML responses from gateways
            msg = str(e)
            # Retry with alternate token parameter if provider rejects 'max_tokens'
            if "Unsupported parameter" in msg and "max_tokens" in msg:
                try:
                    _breadcrumb("llm:param-fallback:max_completion_tokens")
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = max_tokens
                    resp = client.chat.completions.create(**kwargs)
                    _breadcrumb("llm:chat.create:after")
                    out = resp.choices[0].message.content or ""
                    _breadcrumb("llm:exit")
                    return out
                except Exception:
                    pass
            # Retry if provider rejects temperature values (some only allow default of 1)
            if ("Unsupported value" in msg or "unsupported_value" in msg) and "temperature" in msg:
                try:
                    _breadcrumb("llm:param-fallback:temperature-default")
                    # Some providers require omitting temperature or forcing 1
                    temp_prev = kwargs.pop("temperature", None)
                    try:
                        resp = client.chat.completions.create(**kwargs)
                    except Exception:
                        # Force to 1 and retry
                        kwargs["temperature"] = 1
                        resp = client.chat.completions.create(**kwargs)
                    _breadcrumb("llm:chat.create:after")
                    out = resp.choices[0].message.content or ""
                    _breadcrumb("llm:exit")
                    return out
                except Exception:
                    # Restore original temperature for any outer retries
                    if "temperature" not in kwargs and temp_prev is not None:
                        kwargs["temperature"] = temp_prev
                    pass
            if os.getenv("GW_LLM_FALLBACK_ON_404", "0") == "1" and ("404" in msg or "NotFound" in msg):
                _breadcrumb("llm:404-fallback:mock")
                head = (prompt[:220] + "...") if len(prompt) > 220 else prompt
                return f"[MOCK LLM RESPONSE AFTER 404]\nSystem: {system or 'n/a'}\nModel: {model_name}\n---\n{head}"
            raise

    return with_backoff(_do_call)
