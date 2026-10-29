"""Try the whole flow: plan -> research -> verify -> coverage (loop back if thin) -> write.

    uv run python -m test.try_full_run
    uv run python -m test.try_full_run "Your own research question"
"""

import sys

from deepdive import db
from deepdive.graph import run_research

DEFAULT_QUESTION = "Which companies lead global EV battery manufacturing, and how large are their market shares?"


def show_progress(step: str, state: dict) -> None:
    """Called after each step of the graph."""
    if step == "plan":
        print(f"\n[plan] {len(state['sub_questions'])} sub-questions")
    elif step == "research":
        print(f"[research] {len(state['sources'])} sources, {len(state['claims'])} claims so far")
    elif step == "verify":
        passed = sum(c["verdict"] == "supported" for c in state["claims"])
        print(f"[verify] {passed} of {len(state['claims'])} claims passed")
    elif step == "coverage":
        print("[coverage] " + ", ".join(f"{sq['id']}={sq['status']}" for sq in state["sub_questions"]))
    elif step == "write":
        print("[write] report done")


question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
print(f"Question: {question}")
state = run_research(question, on_step=show_progress)

print("\n" + "=" * 80)
print(state["report"])

print("=" * 80)
print("AUDIT LOG")
for entry in db.get_run(state["run_id"])["log"]:
    print(f"  {entry['step']:<9} {entry['message']}")
