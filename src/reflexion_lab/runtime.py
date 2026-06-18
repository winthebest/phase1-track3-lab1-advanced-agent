"""Chọn runtime: Gemini thật hay mock.

Ưu tiên mock nếu:
- USE_MOCK=1, hoặc
- không có GEMINI_API_KEY/GOOGLE_API_KEY (để autograde/CI chạy được offline).

Export cùng API cho agents.py: actor_answer, evaluator, reflector,
FAILURE_MODE_BY_QID, và cờ IS_MOCK.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# FAILURE_MODE_BY_QID chỉ phục vụ mock; với dữ liệu thật agents.py sẽ tự phân loại.
from .mock_runtime import FAILURE_MODE_BY_QID


def _use_mock() -> bool:
    if os.getenv("USE_MOCK", "0") == "1":
        return True
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return True
    return False


IS_MOCK = _use_mock()

if IS_MOCK:
    from .mock_runtime import actor_answer, evaluator, reflector
else:
    from .gemini_runtime import actor_answer, evaluator, reflector

__all__ = [
    "actor_answer",
    "evaluator",
    "reflector",
    "FAILURE_MODE_BY_QID",
    "IS_MOCK",
]
