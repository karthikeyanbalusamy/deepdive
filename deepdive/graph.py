"""The coordinator: a LangGraph that passes work between the agents.

    plan -> research -> verify -> coverage --(some sub-question thin, retries left)--> research
                                           |
                                           +--(all done)----------------------------> write -> END

Each step reads the shared state, does its job, logs what it did, and returns the fields it changed.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from deepdive import agents, config, db, report, rules
from deepdive.llm import QuotaExceeded


class State(TypedDict, total=False):
    run_id: str
    question: str
    sub_questions: list[dict]  # {id, text, status: todo | covered | thin, attempts, gap}
    sources: list[dict]        # {id, sub_q, url, title, text}
    claims: list[dict]         # {id, sub_q, text, source_id, verdict, quote, reason}
    conflicts: list[str]       # where sources disagree
    sentences: list[dict]      # final report sentences with the sources they cite
    report: str                # final Markdown report


def plan_step(state: State) -> dict:
    texts = agents.plan(state["question"])
    sub_questions = [{"id": f"Q{i}", "text": t, "status": "todo", "attempts": 0, "gap": ""} for i, t in enumerate(texts, 1)]
    for sq in sub_questions:
        db.log(state["run_id"], "plan", f"{sq['id']}: {sq['text']}")
    return {"sub_questions": sub_questions, "sources": [], "claims": [], "conflicts": []}


def research_one(run_id: str, sq: dict) -> tuple[list[dict], list, list[str]]:
    """Research one sub-question. Returns (sources, claims, queries); empty lists if it failed."""
    try:
        return agents.research(sq["text"], sq["gap"])
    except QuotaExceeded:  # no point continuing: every later model call would fail too
        raise
    except Exception as exc:  # one failed sub-question must not stop the run; coverage will retry it
        db.log(run_id, "research", f"{sq['id']} failed: {exc}")
        return [], [], []


def research_step(state: State) -> dict:
    run_id = state["run_id"]
    todo = [sq for sq in state["sub_questions"] if sq["status"] == "todo"]
    with ThreadPoolExecutor() as pool:  # sub-questions are independent, so research them all at the same time
        results = dict(zip([sq["id"] for sq in todo], pool.map(lambda sq: research_one(run_id, sq), todo)))

    sources, claims, sub_questions = list(state["sources"]), list(state["claims"]), []
    by_url = {source["url"]: i for i, source in enumerate(sources)}  # position of each web page in `sources`
    for sq in state["sub_questions"]:
        if sq["id"] in results:
            sq = {**sq, "attempts": sq["attempts"] + 1}
            found_sources, found_claims, queries = results[sq["id"]]
            for query in queries:
                db.log(run_id, "research", f'{sq["id"]} searched "{query}"')

            # Each researcher numbers its sources S1, S2, ...; give them ids that are unique in the whole run.
            # A web page found again keeps its first id; any new text from it is added to that source.
            new_ids = {}
            for source in found_sources:
                if source["url"] in by_url:
                    i = by_url[source["url"]]
                    if source["text"] not in sources[i]["text"]:
                        sources[i] = {**sources[i], "text": sources[i]["text"] + "\n\n" + source["text"]}
                    new_ids[source["id"]] = sources[i]["id"]
                else:
                    new_ids[source["id"]] = f"S{len(sources) + 1}"
                    by_url[source["url"]] = len(sources)
                    sources.append({**source, "id": new_ids[source["id"]], "sub_q": sq["id"]})

            for claim in found_claims:
                if claim.source_id not in new_ids:
                    db.log(run_id, "research", f"Dropped claim citing unknown source {claim.source_id}: {claim.text}")
                    continue
                claims.append({"id": f"C{len(claims) + 1}", "sub_q": sq["id"], "text": claim.text,
                               "source_id": new_ids[claim.source_id], "verdict": None, "quote": "", "reason": ""})
            db.log(run_id, "research", f"{sq['id']}: {len(found_claims)} claim(s) from {len(found_sources)} source(s)")
        sub_questions.append(sq)
    return {"sub_questions": sub_questions, "sources": sources, "claims": claims}


def verify_batch(run_id: str, batch: list[dict], sources: dict[str, dict]) -> agents.Verification:
    """Verify the claims of one sub-question. If the verifier fails, no claim gets a verdict."""
    try:
        return agents.verify(batch, sources)
    except QuotaExceeded:
        raise
    except Exception as exc:  # claims then stay unverified rather than slipping through
        db.log(run_id, "verify", f"Verifier failed for {batch[0]['sub_q']}: {exc}")
        return agents.Verification(checks=[])


def verify_step(state: State) -> dict:
    run_id = state["run_id"]
    sources = {s["id"]: s for s in state["sources"]}
    unchecked = [c for c in state["claims"] if c["verdict"] is None]
    batches = [[c for c in unchecked if c["sub_q"] == sq_id] for sq_id in dict.fromkeys(c["sub_q"] for c in unchecked)]
    with ThreadPoolExecutor() as pool:  # one verifier call per sub-question, all at the same time
        results = list(pool.map(lambda batch: verify_batch(run_id, batch, sources), batches))

    checked, conflicts = {}, list(state["conflicts"])
    for batch, result in zip(batches, results):
        conflicts += result.conflicts
        checks = {check.claim_id: check for check in result.checks}
        for claim in batch:
            check = checks.get(claim["id"])
            if check is None:
                verdict, quote, reason = "unsupported", "", "verifier gave no verdict"
            elif check.verdict != "unsupported" and not rules.quote_in_source(check.quote, sources[claim["source_id"]]["text"]):
                verdict, quote, reason = "unsupported", check.quote, f"quote not found in source (verifier said {check.verdict})"
            else:
                verdict, quote, reason = check.verdict, check.quote, check.reason
            checked[claim["id"]] = {**claim, "verdict": verdict, "quote": quote, "reason": reason}
            label = "PASSED" if verdict == "supported" else "REJECTED"
            db.log(run_id, "verify", f"{label} {claim['id']} ({verdict}): {claim['text']} | {reason}")

    for conflict in conflicts[len(state["conflicts"]):]:
        db.log(run_id, "verify", f"Sources disagree: {conflict}")
    claims = [checked.get(c["id"], c) for c in state["claims"]]
    return {"claims": claims, "conflicts": conflicts}


def coverage_step(state: State) -> dict:
    """Plain code, no LLM: is each sub-question well supported, or must it be researched again?"""
    run_id = state["run_id"]
    sources = {s["id"]: s for s in state["sources"]}
    sub_questions = []
    for sq in state["sub_questions"]:
        if sq["status"] == "todo":
            own_claims = [c for c in state["claims"] if c["sub_q"] == sq["id"]]
            gap = rules.coverage_gap(own_claims, sources)
            if not gap:
                sq = {**sq, "status": "covered", "gap": ""}
                db.log(run_id, "coverage", f"{sq['id']} is well supported")
            elif sq["attempts"] <= config.MAX_RETRIES:
                rejected = [f'"{c["text"]}" was rejected: {c["reason"]}' for c in own_claims if c["verdict"] != "supported"]
                sq = {**sq, "gap": "; ".join([gap] + rejected[:3])}
                db.log(run_id, "coverage", f"{sq['id']} is thin ({gap}), researching again")
            else:
                sq = {**sq, "status": "thin", "gap": gap}
                db.log(run_id, "coverage", f"{sq['id']} is still thin after {sq['attempts']} attempts ({gap})")
        sub_questions.append(sq)
    return {"sub_questions": sub_questions}


def next_step(state: State) -> str:
    return "research" if any(sq["status"] == "todo" for sq in state["sub_questions"]) else "write"


def write_step(state: State) -> dict:
    verified = [c for c in state["claims"] if c["verdict"] == "supported"]
    draft = agents.write(state["question"], state["sub_questions"], verified) if verified else None
    markdown, sentences, dropped = report.render(state, draft)
    for text in dropped:
        db.log(state["run_id"], "write", f"Dropped sentence without verified citation: {text}")
    db.log(state["run_id"], "write", f"Report written: {len(sentences)} sentence(s), {len(dropped)} dropped")
    return {"report": markdown, "sentences": sentences}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("plan", plan_step)
    graph.add_node("research", research_step)
    graph.add_node("verify", verify_step)
    graph.add_node("coverage", coverage_step)
    graph.add_node("write", write_step)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "verify")
    graph.add_edge("verify", "coverage")
    graph.add_conditional_edges("coverage", next_step, ["research", "write"])
    graph.add_edge("write", END)
    return graph.compile()


def run_research(question: str, on_step=None) -> dict:
    """Run the whole research for a question and return the final state.

    `on_step(step_name, state)` is called after each step, so a UI can show progress.
    """
    run_id = db.create_run(question)
    state = {"run_id": run_id, "question": question}
    try:
        for update in build_graph().stream(state, {"recursion_limit": 50}):
            for step, changes in update.items():
                state.update(changes)
                if on_step:
                    on_step(step, state)
    except Exception as exc:
        db.log(run_id, "error", f"Run failed: {exc}")
        db.fail_run(run_id, str(exc))
        raise
    db.finish_run(run_id, state)
    return state
