"""Try the researcher agent on its own: it searches the web for one sub-question and drafts claims.

    uv run python -m test.try_researcher
    uv run python -m test.try_researcher "Your own sub-question"
"""

import sys

from deepdive import config
from deepdive.agents import research

DEFAULT_SUB_QUESTION = "Which companies lead global EV battery manufacturing by market share in 2025?"

sub_question = " ".join(sys.argv[1:]) or DEFAULT_SUB_QUESTION
print(f"Model:        {config.MODEL}")
print(f"Sub-question: {sub_question}\n")

sources, claims, queries = research(sub_question)

print("Searches:")
for query in queries:
    print(f"  - {query}")

print(f"\nSources found ({len(sources)}):")
for source in sources:
    print(f"  [{source['id']}] {source['title']}")
    print(f"       {source['url']}")

print(f"\nClaims drafted ({len(claims)}):")
for claim in claims:
    print(f"  - {claim.text}  [cites {claim.source_id}]")
