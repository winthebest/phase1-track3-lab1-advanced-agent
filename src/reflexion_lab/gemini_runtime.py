"""Runtime thật dùng Gemini — thay thế mock_runtime.

Cùng signature với mock_runtime để agents.py dùng được không đổi:
- actor_answer(example, attempt_id, agent_type, reflection_memory) -> str
- evaluator(example, answer) -> JudgeResult
- reflector(example, attempt_id, judge) -> ReflectionEntry
"""
from __future__ import annotations
from pydantic import BaseModel

from .llm import generate_text, generate_json
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, ReflectionEntry, QAExample


# Response schema riêng cho LLM (Gemini-friendly: tránh Literal, bỏ field code tự gán)
class _JudgeResponse(BaseModel):
    score: int
    reason: str
    missing_evidence: list[str] = []
    spurious_claims: list[str] = []


class _ReflectionResponse(BaseModel):
    failure_reason: str
    lesson: str
    next_strategy: str


def _format_context(example: QAExample) -> str:
    return "\n".join(f"- {c.title}: {c.text}" for c in example.context)


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    parts = [
        f"Question: {example.question}",
        "",
        "Context paragraphs:",
        _format_context(example),
    ]
    if reflection_memory:
        parts += ["", "Reflection notes from previous failed attempts:"]
        parts += [f"- {note}" for note in reflection_memory]
    parts += ["", "Final answer:"]
    result = generate_text(ACTOR_SYSTEM, "\n".join(parts))
    return result.text


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    user = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    resp, _ = generate_json(EVALUATOR_SYSTEM, user, _JudgeResponse)
    score = 1 if int(resp.score) == 1 else 0
    return JudgeResult(
        score=score,
        reason=resp.reason,
        missing_evidence=resp.missing_evidence,
        spurious_claims=resp.spurious_claims,
    )


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    user = (
        f"Question: {example.question}\n"
        f"Context paragraphs:\n{_format_context(example)}\n\n"
        f"Evaluator reason it was wrong: {judge.reason}\n"
        f"Wrong/unsupported claims: {judge.spurious_claims}\n"
        f"Missing evidence: {judge.missing_evidence}"
    )
    resp, _ = generate_json(REFLECTOR_SYSTEM, user, _ReflectionResponse)
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=resp.failure_reason,
        lesson=resp.lesson,
        next_strategy=resp.next_strategy,
    )
