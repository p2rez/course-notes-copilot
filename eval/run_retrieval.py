import json
from pathlib import Path

from src.hybrid import hybrid_search
from src.search import search

K = 5


def load_questions(path: str = "eval/questions.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(result: dict, q: dict) -> bool:
    return result["source"] == q["source"] and result["page"] in q["pages"]


def evaluate(name: str, search_fn, strategy: str, questions: list[dict]) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    rr_total = 0.0
    for q in questions:
        results = search_fn(q["question"], strategy=strategy, k=K)
        rank = next((i for i, r in enumerate(results, start=1) if is_hit(r, q)), None)
        if rank:
            rr_total += 1 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
    n = len(questions)
    return {
        "retriever": name,
        "chunking": strategy,
        "n": n,
        **{f"hit@{k}": round(v / n, 3) for k, v in hits.items()},
        "mrr": round(rr_total / n, 3),
    }


if __name__ == "__main__":
    questions = load_questions()
    retrievers = {"dense": search, "hybrid": hybrid_search}
    rows = [
        evaluate(name, fn, s, questions)
        for name, fn in retrievers.items()
        for s in ["fixed", "page"]
    ]
    print(f"{'retriever':9}{'chunking':9}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'mrr':>7}")
    for r in rows:
        print(f"{r['retriever']:9}{r['chunking']:9}{r['hit@1']:>7}{r['hit@3']:>7}{r['hit@5']:>7}{r['mrr']:>7}")
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/retrieval_results.json").write_text(json.dumps(rows, indent=2))
