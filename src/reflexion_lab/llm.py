"""Gemini client wrapper cho lab (model mặc định: gemini-2.5-flash-lite).

Cung cấp:
- generate_text(system, user)            -> LLMResult (text + token + latency)
- generate_json(system, user, schema)    -> (parsed_pydantic, LLMResult)

Đồng thời tích lũy token/latency vào một accumulator toàn cục để Bước 5
(agents.py) lấy số liệu thật cho mỗi attempt: reset_usage() / collect_usage().
"""
from __future__ import annotations
import os
import time
from functools import lru_cache
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types, errors

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
# Tắt "thinking" mặc định: model 2.5 bật thinking sinh hàng chục nghìn token ẩn,
# đội token + latency. Lab QA chỉ cần đáp án ngắn nên budget=0. Đổi qua env nếu cần.
THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))


def _thinking_config():
    return types.ThinkingConfig(thinking_budget=THINKING_BUDGET)

T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


# --- Accumulator: tổng token/latency các call kể từ lần reset gần nhất ---
_ACC = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency_ms": 0}


def reset_usage() -> None:
    for k in _ACC:
        _ACC[k] = 0


def collect_usage() -> dict[str, int]:
    return dict(_ACC)


def _accumulate(r: LLMResult) -> None:
    _ACC["prompt_tokens"] += r.prompt_tokens
    _ACC["output_tokens"] += r.output_tokens
    _ACC["total_tokens"] += r.total_tokens
    _ACC["latency_ms"] += r.latency_ms


@lru_cache(maxsize=1)
def _client() -> "genai.Client":
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY chưa được set. Tạo file .env với GEMINI_API_KEY=... "
            "hoặc chạy với USE_MOCK=1 để dùng mock runtime."
        )
    return genai.Client(api_key=key)


MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "6"))
# Mã lỗi tạm thời nên retry: 429 (rate limit), 500/503 (server quá tải).
_RETRYABLE = {429, 500, 502, 503, 504}


def _generate_with_retry(user: str, config: "types.GenerateContentConfig"):
    """Gọi Gemini với exponential backoff cho lỗi tạm thời (503/429/...)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _client().models.generate_content(model=MODEL, contents=user, config=config)
        except (errors.ServerError, errors.ClientError) as exc:
            code = getattr(exc, "code", None)
            if code not in _RETRYABLE or attempt == MAX_RETRIES - 1:
                raise
            last_exc = exc
            time.sleep(min(2 ** attempt, 30))  # 1,2,4,8,16,30s...
    if last_exc:
        raise last_exc


def _usage(resp) -> tuple[int, int, int]:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return 0, 0, 0
    return (
        u.prompt_token_count or 0,
        u.candidates_token_count or 0,
        u.total_token_count or 0,
    )


def generate_text(system: str, user: str, temperature: float = 0.0) -> LLMResult:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        thinking_config=_thinking_config(),
    )
    t0 = time.perf_counter()
    resp = _generate_with_retry(user, config)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    p, o, tot = _usage(resp)
    result = LLMResult(
        text=(resp.text or "").strip(),
        prompt_tokens=p,
        output_tokens=o,
        total_tokens=tot,
        latency_ms=latency_ms,
    )
    _accumulate(result)
    return result


def generate_json(system: str, user: str, schema: Type[T], temperature: float = 0.0) -> tuple[T, LLMResult]:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=_thinking_config(),
    )
    t0 = time.perf_counter()
    resp = _generate_with_retry(user, config)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    p, o, tot = _usage(resp)
    result = LLMResult(
        text=(resp.text or "").strip(),
        prompt_tokens=p,
        output_tokens=o,
        total_tokens=tot,
        latency_ms=latency_ms,
    )
    _accumulate(result)
    parsed = resp.parsed
    if parsed is None:
        # Fallback: tự parse JSON nếu SDK không tự bind được
        parsed = schema.model_validate_json(result.text)
    return parsed, result
