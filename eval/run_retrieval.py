import json
from pathlib import Path

from src.search import search

K = 5


def load_questions(path: str = "eval/questions.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(result: dict, q: dict) -> bool:
    return result["source"] == q["source"] and result["page"] in q["pages"]


def evaluate(strategy: str, questions: list[dict]) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    rr_total = 0.0
    for q in questions:
        results = search(q["question"], strategy=strategy, k=K)
        rank = next((i for i, r in enumerate(results, start=1) if is_hit(r, q)), None)
        if rank:
            rr_total += 1 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
    n = len(questions)
    return {
        "strategy": strategy,
        "n": n,
        **{f"hit@{k}": round(v / n, 3) for k, v in hits.items()},
        "mrr": round(rr_total / n, 3),
    }


if __name__ == "__main__":
    questions = load_questions()
    rows = [evaluate(s, questions) for s in ["fixed", "page"]]
    for r in rows:
        print(r)
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/dense_baseline.json").write_text(json.dumps(rows, indent=2))
