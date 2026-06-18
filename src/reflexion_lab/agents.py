from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .runtime import FAILURE_MODE_BY_QID, IS_MOCK, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer

# Số đo token/latency thật chỉ có khi chạy LLM (Gemini). Khi mock thì dùng
# công thức ước lượng cũ để report mock vẫn có số liệu hợp lý.
if not IS_MOCK:
    from .llm import collect_usage, reset_usage


def classify_failure_mode(traces: list[AttemptTrace], final_judge: JudgeResult | None) -> str:
    """Phân loại loại lỗi từ tín hiệu của evaluator (thay cho lookup hardcode).

    - none                 : trả lời đúng
    - looping              : nhiều attempt nhưng lặp lại cùng một đáp án sai
    - incomplete_multi_hop : evaluator báo còn thiếu bằng chứng/hop
    - entity_drift         : evaluator báo có khẳng định sai/lạc thực thể
    - wrong_final_answer   : sai nhưng không rơi vào các nhóm trên
    """
    if final_judge is None or final_judge.score == 1:
        return "none"
    answers = [normalize_answer(t.answer) for t in traces]
    if len(answers) >= 2 and len(set(answers)) == 1:
        return "looping"
    if final_judge.missing_evidence:
        return "incomplete_multi_hop"
    if final_judge.spurious_claims:
        return "entity_drift"
    return "wrong_final_answer"

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_judge: JudgeResult | None = None
        for attempt_id in range(1, self.max_attempts + 1):
            if not IS_MOCK:
                reset_usage()
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            final_answer = answer
            final_score = judge.score
            final_judge = judge

            # Reflexion: sau một lần thử sai, phản chiếu để lần sau tốt hơn.
            # Chỉ reflect khi còn lượt thử kế tiếp (attempt_id < max_attempts) — reflect ở lượt cuối là lãng phí.
            entry: ReflectionEntry | None = None
            if judge.score != 1 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                entry = reflector(example, attempt_id, judge)
                reflections.append(entry)
                reflection_memory.append(
                    f"[Attempt {entry.attempt_id}] Lý do sai: {entry.failure_reason} "
                    f"Bài học: {entry.lesson} Lần sau: {entry.next_strategy}"
                )

            # Token & latency: số thật từ LLM (gồm actor + evaluator + reflector của attempt này),
            # fallback công thức ước lượng khi chạy mock.
            if IS_MOCK:
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
                prompt_tokens = output_tokens = 0
            else:
                u = collect_usage()
                token_estimate = u["total_tokens"]
                prompt_tokens = u["prompt_tokens"]
                output_tokens = u["output_tokens"]
                latency_ms = u["latency_ms"]

            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason,
                                 token_estimate=token_estimate, prompt_tokens=prompt_tokens,
                                 output_tokens=output_tokens, latency_ms=latency_ms, reflection=entry)
            traces.append(trace)
            if judge.score == 1:
                break
        total_tokens = sum(t.token_estimate for t in traces)
        total_prompt = sum(t.prompt_tokens for t in traces)
        total_output = sum(t.output_tokens for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        # Mock: giữ loại lỗi đã thiết kế sẵn theo qid. Real: phân loại từ evaluator.
        if IS_MOCK:
            failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        else:
            failure_mode = classify_failure_mode(traces, final_judge)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, prompt_tokens=total_prompt, output_tokens=total_output, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
