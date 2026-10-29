"""DeepDive web page.  Run with:  uv run streamlit run ui.py"""

import streamlit as st

from deepdive import db, report
from deepdive.graph import run_research

STEP_LABELS = {
    "plan": "Planner split the question into sub-questions",
    "research": "Researchers gathered evidence from the web",
    "verify": "Verifier checked every claim against its source",
    "coverage": "Checked whether each sub-question has enough proof",
    "write": "Writer produced the report from verified claims",
}
VERDICT_ICONS = {"supported": "✅", "unsupported": "❌", "contradicted": "⛔"}

st.set_page_config(page_title="DeepDive", page_icon="🔎", layout="wide")
st.title("🔎 DeepDive")
st.caption("A research analyst that checks every claim against its source before it reaches the report.")

# --- ask a question --------------------------------------------------------------------------
question = st.text_input("Research question", placeholder="What are the main risks of investing in the EV battery market in 2026?")
if st.button("Research", type="primary", disabled=len(question.strip()) < 10):
    with st.status("Researching…", expanded=True) as status:
        try:
            state = run_research(question.strip(), on_step=lambda step, _: st.write(f"✓ {STEP_LABELS[step]}"))
            st.session_state.run_id = state["run_id"]
            status.update(label="Done", state="complete")
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Research failed: {exc}")

# --- earlier runs ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Earlier runs")
    for run in db.list_runs():
        if st.button(f"{run['question'][:60]} ({run['status']})", key=run["id"]):
            st.session_state.run_id = run["id"]

# --- show a run ------------------------------------------------------------------------------
run = db.get_run(st.session_state["run_id"]) if "run_id" in st.session_state else None
if run and run["status"] == "failed":
    st.error(f"This run failed: {run['error']}")
elif run and run["state"]:
    state = run["state"]
    claims = state["claims"]
    passed = sum(c["verdict"] == "supported" for c in claims)
    retries = sum(sq["attempts"] - 1 for sq in state["sub_questions"])
    st.subheader(run["question"])
    st.write(f"**{passed}** of **{len(claims)}** claims passed verification · "
             f"**{len(claims) - passed}** rejected · **{retries}** extra research round(s)")

    tab_report, tab_claims, tab_log = st.tabs(["📄 Report", "🧾 Claims checked", "📜 Log"])
    with tab_report:
        st.markdown(run["report"])
        col_html, col_md = st.columns(2)
        col_html.download_button("Download report (HTML: open in any browser, print to PDF)",
                                 report.to_html(run["report"], run["question"]),
                                 file_name=f"deepdive-{run['id']}.html", mime="text/html")
        col_md.download_button("Download report (Markdown)", run["report"], file_name=f"deepdive-{run['id']}.md")
    with tab_claims:
        sources = {s["id"]: s for s in state["sources"]}
        for c in claims:
            with st.expander(f"{VERDICT_ICONS[c['verdict']]} {c['id']}: {c['text']}"):
                source = sources[c["source_id"]]
                st.markdown(f"**Verdict:** {c['verdict']}  \n**Why:** {c['reason']}  \n"
                            f"**Source:** [{source['title']}]({source['url']})")
                if c["quote"]:
                    st.markdown(f"> {c['quote']}")
    with tab_log:
        start = run["log"][0]["time"] if run["log"] else 0
        st.code("\n".join(f"{e['time'] - start:6.1f}s  {e['step']:<9} {e['message']}" for e in run["log"]), language=None)
