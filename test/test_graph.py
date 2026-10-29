"""A full run with scripted fake models and fake search: real create_agent + LangGraph, no network."""

import itertools
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from deepdive import agents, config, db, graph, llm, search
from deepdive.graph import run_research
from deepdive.llm import QuotaExceeded

CALL_IDS = itertools.count()
PAGE1 = {"url": "https://www.reuters.com/a", "title": "Reuters", "text": "CATL held 37.9% of the global EV battery market in 2024."}
PAGE2 = {"url": "https://www.iea.org/b", "title": "IEA", "text": "BYD ranked second with a 17.2% share of the market in 2024."}


def tool_call(name: str, **args) -> AIMessage:
    """A model reply that calls a tool, e.g. web_search."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{next(CALL_IDS)}"}])


def answer(**fields) -> AIMessage:
    """A model reply that is the agent's final answer, as JSON."""
    return AIMessage(content=json.dumps(fields))


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def bind(self, **kwargs):
        return self


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(search, "web_search", lambda query: [next(pages)])
    pages = iter([PAGE1, PAGE2])
    replies = []
    shared = iter(replies)  # agents run one after another, so one script in call order is enough
    monkeypatch.setattr(llm, "chat_model", lambda model: FakeModel(messages=shared))
    return replies


def test_thin_sub_question_loops_back_and_unverified_text_is_dropped(fake_world):
    fake_world += [
        answer(sub_questions=["Who leads EV batteries?"]),
        # attempt 1: one search, one real claim and one citing a source that was never retrieved
        tool_call("web_search", query="EV battery market share 2024"),
        answer(claims=[{"text": "CATL held 37.9% in 2024", "source_id": "S1"},
                {"text": "Invented fact", "source_id": "S9"}]),
        answer(checks=[{"claim_id": "C1", "verdict": "supported", "reason": "stated",
                        "quote": "CATL held 37.9% of the global EV battery market"}]),
        # coverage: only 1 verified claim -> research again
        tool_call("web_search", query="BYD EV battery share 2024"),
        # each researcher numbers its own sources from S1; the coordinator renames this one to S2
        answer(claims=[{"text": "BYD was second with 17.2%", "source_id": "S1"}]),
        answer(checks=[{"claim_id": "C2", "verdict": "supported", "reason": "stated",
                        "quote": "BYD ranked second with a 17.2% share"}]),
        answer(title="EV batteries", summary=[
            {"text": "CATL leads and BYD is second.", "claim_ids": ["C1", "C2"]},
            {"text": "A sentence the model made up.", "claim_ids": []},
        ], sections=[]),
    ]

    state = run_research("Who leads the EV battery market?")

    assert [c["verdict"] for c in state["claims"]] == ["supported", "supported"]
    assert state["sub_questions"][0]["status"] == "covered" and state["sub_questions"][0]["attempts"] == 2
    assert "CATL leads and BYD is second. [S1][S2]" in state["report"]
    assert "made up" not in state["report"]

    run = db.get_run(state["run_id"])
    log = "\n".join(e["message"] for e in run["log"])
    assert run["status"] == "done"
    assert "Dropped claim citing unknown source S9" in log
    assert "researching again" in log
    assert "Dropped sentence without verified citation" in log


def test_run_stops_immediately_when_daily_limit_is_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(agents, "plan", lambda question: ["Q one?", "Q two?"])
    calls = []

    def research_out_of_quota(*args):
        calls.append(args)
        raise QuotaExceeded("Daily model limit reached.")

    monkeypatch.setattr(agents, "research", research_out_of_quota)
    with pytest.raises(QuotaExceeded):
        run_research("Who leads the EV battery market?")

    assert len(calls) == 1  # stopped at the first failure, no further sub-questions or retry rounds
    run = db.list_runs()[0]
    assert run["status"] == "failed"


def test_same_web_page_found_twice_becomes_one_source(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    page = "https://finance.yahoo.com/news/sodium"
    found = {
        "Q1?": ([{"id": "S1", "url": page, "title": "Yahoo", "text": "Market was USD 324 million in 2023."}],
                [agents.Claim(text="USD 324 million in 2023", source_id="S1")], []),
        "Q2?": ([{"id": "S1", "url": "https://iea.org/x", "title": "IEA", "text": "Other page."},
                 {"id": "S2", "url": page, "title": "Yahoo", "text": "CATL and HiNa are key players."}],
                [agents.Claim(text="CATL and HiNa are key players", source_id="S2")], []),
    }
    monkeypatch.setattr(agents, "research", lambda text, gap: found[text])
    state = {"run_id": db.create_run("q"), "sources": [], "claims": [], "sub_questions": [
        {"id": "Q1", "text": "Q1?", "status": "todo", "attempts": 0, "gap": ""},
        {"id": "Q2", "text": "Q2?", "status": "todo", "attempts": 0, "gap": ""},
    ]}

    result = graph.research_step(state)

    assert [s["url"] for s in result["sources"]] == [page, "https://iea.org/x"]  # the Yahoo page appears once
    assert result["sources"][0]["text"] == "Market was USD 324 million in 2023.\n\nCATL and HiNa are key players."
    assert [c["source_id"] for c in result["claims"]] == ["S1", "S1"]  # both claims point to that one source


def test_run_fails_clearly_when_web_search_is_out_of_credits(fake_world, monkeypatch):
    def search_out_of_credits(query):
        raise QuotaExceeded("Web search limit reached (Tavily).")

    monkeypatch.setattr(search, "web_search", search_out_of_credits)
    fake_world += [
        answer(sub_questions=["When did JWST launch?"]),
        tool_call("web_search", query="JWST launch date"),
        answer(claims=[]),
    ]
    with pytest.raises(QuotaExceeded, match="Web search limit"):
        run_research("When did the James Webb Space Telescope launch?")

    run = db.list_runs()[0]
    assert run["status"] == "failed"  # not "done" with an empty "not enough evidence" report
