"""The four agents. Each has its own job, its own prompt and its own output format.

    planner     question         -> sub-questions
    researcher  sub-question     -> web sources + claims citing them   (the only agent with a tool)
    verifier    claims + sources -> a verdict for each claim
    writer      verified claims  -> report sentences, each citing the claims it uses
"""

import re
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from deepdive import config, search
from deepdive.llm import QuotaExceeded, run_agent

# --- Planner ----------------------------------------------------------------------------------


class Plan(BaseModel):
    sub_questions: list[str]


PLANNER_PROMPT = f"""You plan research. Split the user's question into at most {config.MAX_SUB_QUESTIONS}
focused, non-overlapping sub-questions that together answer it (for example: market size, key players,
risks, recent events). Each sub-question must be answerable with a web search."""


def plan(question: str) -> list[str]:
    result = run_agent(config.MODEL, PLANNER_PROMPT, Plan, question)
    return result.sub_questions[: config.MAX_SUB_QUESTIONS]


# --- Researcher -------------------------------------------------------------------------------


class Claim(BaseModel):
    text: str = Field(description="one short factual statement")
    source_id: str = Field(description="id of the search result it comes from, e.g. S3")


class Findings(BaseModel):
    claims: list[Claim]


RESEARCHER_PROMPT = f"""You research ONE sub-question using the web_search tool.
- Search with specific queries (names, years, numbers). You may search up to {config.MAX_SEARCHES} times.
- Then list up to 6 factual claims that answer the sub-question.
- Every claim must come from one search result and cite its id (e.g. S3).
- Copy numbers, dates and names exactly as written in the source. Never add your own knowledge.
- Prefer claims from different websites. If nothing useful was found, return no claims.
An independent verifier will reject any claim that its source does not support."""


def research(sub_question: str, feedback: str = "") -> tuple[list[dict], list[Claim], list[str]]:
    """Research one sub-question. Returns (sources found, claims, queries searched).

    Sources are numbered S1, S2, ... within this call; the coordinator makes them unique across the run.
    """
    sources: list[dict] = []
    queries: list[str] = []
    out_of_credits: list[QuotaExceeded] = []  # raised after the agent finishes, so the whole run stops

    @tool
    def web_search(query: str) -> str:
        """Search the web. Returns results with ids (S1, S2, ...) that claims must cite."""
        if out_of_credits:
            return "Web search is unavailable. Return no claims."
        if len(queries) >= config.MAX_SEARCHES:
            return "Search limit reached. Return your claims now."
        queries.append(query)
        try:
            results = search.web_search(query)
        except QuotaExceeded as exc:
            out_of_credits.append(exc)
            return "Web search is unavailable. Return no claims."
        except Exception as exc:  # e.g. a timeout: the researcher can try another query
            return f"Search failed: {exc}"
        lines = []
        for result in results:
            source = {**result, "id": f"S{len(sources) + 1}"}
            sources.append(source)
            lines.append(f"[{source['id']}] {source['title']} ({source['url']})\n{source['text']}")
        return "\n\n".join(lines) or "No results. Try a different query."

    message = f"Sub-question: {sub_question}"
    if feedback:
        message += f"\n\nA previous attempt was not good enough: {feedback}\nSearch again with different queries."
    findings = run_agent(config.MODEL, RESEARCHER_PROMPT, Findings, message, tools=[web_search])
    if out_of_credits:
        raise out_of_credits[0]
    return sources, findings.claims, queries


# --- Verifier ---------------------------------------------------------------------------------


class Check(BaseModel):
    claim_id: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    quote: str = Field(description="the sentence copied word for word from the source, or empty if none")
    reason: str = Field(description="one sentence explaining the verdict")


class Verification(BaseModel):
    checks: list[Check]
    conflicts: list[str] = Field(
        default_factory=list,
        description="disagreements between sources, in words, naming the source ids (e.g. 'S2 says 37%, S5 says 34%')",
    )

    @field_validator("conflicts", mode="before")
    @classmethod
    def no_conflicts_as_empty_list(cls, value):
        """Some models send null instead of [] when there are no conflicts; both mean "none"."""
        return value or []


VERIFIER_PROMPT = """You are a strict fact-checker. For EACH claim, read only the source text given with it
and decide:
- supported: the source clearly states the claim (numbers, dates and names must match)
- unsupported: the source does not state the claim
- contradicted: the source says something different
Copy the quote word for word from the source. If it comes from separate places in the source,
separate the parts with "...". Do not use your own knowledge.
Also list conflicts: places where sources disagree with each other (e.g. different numbers for the same thing)."""


def verify(claims: list[dict], sources: dict[str, dict]) -> Verification:
    message = "\n\n".join(
        f"CLAIM {c['id']}: {c['text']}\nSOURCE {c['source_id']}: {sources[c['source_id']]['text']}" for c in claims
    )
    return run_agent(config.VERIFIER_MODEL, VERIFIER_PROMPT, Verification, message)


# --- Writer -----------------------------------------------------------------------------------


class Sentence(BaseModel):
    text: str
    claim_ids: list[str] = Field(default_factory=list, description="ids of the verified claims this sentence is based on")

    @model_validator(mode="after")
    def take_ids_from_text(self):
        """Some models write the ids into the text, e.g. "... is cheap (C2, C5)." Move them to claim_ids."""
        if not self.claim_ids:
            self.claim_ids = re.findall(r"\bC\d+\b", self.text)
        self.text = re.sub(r"\s*[\(\[]C\d+(?:\s*,\s*C\d+)*[\)\]]", "", self.text).strip()
        return self


class Section(BaseModel):
    heading: str
    sentences: list[Sentence]


class Draft(BaseModel):
    title: str
    summary: list[Sentence]
    sections: list[Section]


WRITER_PROMPT = """You write a research report using ONLY the verified claims given.
- Start with a 2-3 sentence summary, then one section per sub-question.
- Every sentence must list the claim ids it is based on in its claim_ids field (not in the text).
  Sentences without claim ids will be deleted.
- Do not add any fact, number or name that is not in the claims."""


def write(question: str, sub_questions: list[dict], claims: list[dict]) -> Draft:
    lines = [f"Question: {question}"]
    for sq in sub_questions:
        own = [c for c in claims if c["sub_q"] == sq["id"]]
        if own:
            lines.append(f"\nSub-question: {sq['text']}")
            lines += [f"  {c['id']}: {c['text']}" for c in own]
    return run_agent(config.MODEL, WRITER_PROMPT, Draft, "\n".join(lines))
