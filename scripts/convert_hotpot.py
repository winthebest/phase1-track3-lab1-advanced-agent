"""Convert HotpotQA (dev distractor) sang format QAExample của lab.

Nguồn: hotpot_dev_distractor_v1.json (mỗi item: _id, question, answer, level,
type, context=[[title, [sentences...]], ...], supporting_facts=[[title, idx], ...]).

Map sang QAExample:
  qid          <- _id
  difficulty   <- level   (easy/medium/hard)
  question     <- question
  gold_answer  <- answer
  context[]    <- mỗi paragraph -> {title, text=" ".join(sentences)}

Mặc định giữ cả 10 đoạn (gồm distractor). Dùng --only-supporting để chỉ giữ
các đoạn nằm trong supporting_facts (gọn, ít token hơn cho LLM thật).

Chạy:
  python scripts/convert_hotpot.py --num 150
  python scripts/convert_hotpot.py --num 150 --only-supporting
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "hotpot_dev_distractor_v1.json" / "hotpot_dev_distractor_v1.json"
VALID_LEVELS = {"easy", "medium", "hard"}


def convert_item(item: dict, only_supporting: bool) -> dict:
    level = item.get("level", "medium")
    if level not in VALID_LEVELS:
        level = "medium"

    keep_titles = None
    if only_supporting:
        keep_titles = {t for t, _ in item.get("supporting_facts", [])}

    context = []
    for title, sentences in item["context"]:
        if keep_titles is not None and title not in keep_titles:
            continue
        text = " ".join(s.strip() for s in sentences).strip()
        context.append({"title": title, "text": text})

    return {
        "qid": item["_id"],
        "difficulty": level,
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--num", type=int, default=150, help="số mẫu xuất ra")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-supporting", action="store_true",
                    help="chỉ giữ các đoạn supporting_facts (bỏ distractor)")
    ap.add_argument("--out", default=str(ROOT / "data" / "hotpot_dev_qa.json"))
    args = ap.parse_args()

    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    sample = rng.sample(raw, min(args.num, len(raw)))

    examples = [convert_item(it, args.only_supporting) for it in sample]
    Path(args.out).write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    levels = {}
    for e in examples:
        levels[e["difficulty"]] = levels.get(e["difficulty"], 0) + 1
    print(f"Wrote {len(examples)} examples -> {args.out}")
    print(f"levels: {levels}")
    print(f"only_supporting={args.only_supporting}")


if __name__ == "__main__":
    main()
