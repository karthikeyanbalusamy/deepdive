"""Try the planner agent on its own.

    uv run python -m test.try_planner
    uv run python -m test.try_planner "Your own research question"
"""

import sys

from deepdive import config
from deepdive.agents import plan

DEFAULT_QUESTION = "What are the main risks of investing in the EV battery market in 2026?"

question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
print(f"Model:    {config.MODEL}")
print(f"Question: {question}\n")

for number, sub_question in enumerate(plan(question), start=1):
    print(f"{number}. {sub_question}")
