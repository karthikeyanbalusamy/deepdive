"""Self-evaluation: run the test questions and measure how grounded and complete the reports are.

    uv run python -m eval.run_eval                       # all questions
    uv run python -m eval.run_eval known-fact ev-risks   # only some

Metrics (written to eval/results.md):
  pass rate     claims that passed verification / all claims the researchers drafted
  grounded      report sentences that an independent judge model finds supported by their cited sources
  coverage      sub-questions that ended well supported / all sub-questions
  retries       extra research rounds triggered by thin sub-questions
  fact recall   known-answer questions only: expected facts found in the report
"""

import json
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from deepdive import config
from deepdive.graph import run_research
from deepdive.llm import QuotaExceeded, run_agent

HERE = Path(__file__).parent

JUDGE_PROMPT = """You are an independent evaluator. Decide whether the sentence is fully supported by the
source texts. Every number, date and name in the sentence must appear in the sources."""


class Judgement(BaseModel):
    supported: bool
    reason: str


def grounded_sentences(state: dict) -> int:
    """How many report sentences a separate judge model finds supported by their cited sources."""
    sources = {s["id"]: s for s in state["sources"]}
    count = 0
    for sentence in state["sentences"]:
        texts = "\n\n".join(f"[{sid}] {sources[sid]['text']}" for sid in sentence["source_ids"])
        try:
            judgement = run_agent(config.JUDGE_MODEL, JUDGE_PROMPT, Judgement, f"SENTENCE: {sentence['text']}\n\nSOURCES:\n{texts}")
            count += judgement.supported
        except Exception:
            pass  # a failed judgement counts as not supported, so the score is never inflated
    return count


def evaluate(item: dict) -> dict:
    started = time.time()
    try:
        state = run_research(item["question"])
    except QuotaExceeded:  # out of model or search credits: the remaining questions would only record zeros
        raise
    except Exception as exc:
        return {"id": item["id"], "error": str(exc)}

    claims, sub_questions, sentences = state["claims"], state["sub_questions"], state["sentences"]
    passed = sum(c["verdict"] == "supported" for c in claims)
    row = {
        "id": item["id"],
        "category": item["category"],
        "run_id": state["run_id"],  # look up the full log of this question in the web page or GET /runs/{id}
        "claims": len(claims),
        "pass_rate": round(passed / len(claims), 2) if claims else 0.0,
        "grounded": round(grounded_sentences(state) / len(sentences), 2) if sentences else None,
        "coverage": round(sum(sq["status"] == "covered" for sq in sub_questions) / len(sub_questions), 2),
        "retries": sum(sq["attempts"] - 1 for sq in sub_questions),
        "seconds": round(time.time() - started),
    }
    if facts := item.get("expected_facts"):
        row["fact_recall"] = fact_recall(state["report"], facts)
    return row


def fact_recall(report: str, facts: list) -> float:
    """Share of expected facts found in the report.

    A fact is a string, or a list of accepted spellings (e.g. "December 25, 2021" or "25 December 2021").
    """
    report = report.lower()
    found = sum(any(spelling.lower() in report for spelling in ([f] if isinstance(f, str) else f)) for f in facts)
    return round(found / len(facts), 2)


def average(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r.get(key) is not None]
    return f"{statistics.mean(values):.2f}" if values else "-"


def write_results(rows: list[dict]) -> None:
    done = [r for r in rows if "error" not in r]
    columns = ["id", "category", "claims", "pass_rate", "grounded", "coverage", "retries", "fact_recall", "seconds"]
    lines = [
        "# DeepDive self-evaluation",
        "",
        f"Models (via Mesh API): writer/researcher `{config.MODEL}`, verifier `{config.VERIFIER_MODEL}`, judge `{config.JUDGE_MODEL}`.  ",
        f"Run on {time.strftime('%Y-%m-%d')}: {len(done)} of {len(rows)} questions completed.",
        "",
        "| metric | average |",
        "|---|---|",
        f"| verification pass rate | {average(done, 'pass_rate')} |",
        f"| grounded sentences (independent judge) | {average(done, 'grounded')} |",
        f"| sub-question coverage | {average(done, 'coverage')} |",
        f"| extra research rounds | {average(done, 'retries')} |",
        f"| known-answer fact recall | {average(done, 'fact_recall')} |",
        f"| seconds per question | {average(done, 'seconds')} |",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines += ["| " + " | ".join(str(r.get(c, "-")) for c in columns) + " |" for r in done]
    lines += [f"- `{r['id']}` failed: {r['error']}" for r in rows if "error" in r]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    items = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    if only := sys.argv[1:]:
        items = [i for i in items if i["id"] in only]
    rows = []
    for item in items:
        print(f"-> {item['id']}: {item['question']}")
        rows.append(evaluate(item))
        print(f"   {rows[-1]}")
        write_results(rows)  # save after every question so partial results are kept
    print(f"Results written to {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
