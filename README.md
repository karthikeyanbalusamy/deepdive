# DeepDive: a self-verifying research analyst

DeepDive answers a research question with a structured, cited report in which **every claim has been checked
against the source it cites**. Claims the source does not support are removed, thinly supported parts of the
question are researched again, and every step is logged so you can see exactly what was kept, what was rejected,
and why.

Every LLM call goes through **Mesh API** (`https://api.meshapi.ai/v1`). Agents are built with LangChain's
`create_agent` and coordinated by a LangGraph `StateGraph`.

## What it does, end to end

Ask: *"How commercially mature is sodium-ion battery technology compared with lithium-ion?"*

1. The **planner** splits it into sub-questions: market status, key players, pros and cons versus lithium-ion,
   recent developments.
2. A **researcher** per sub-question searches the web, stores each result as a numbered source (S1, S2, …) with
   its URL and text, and drafts short factual claims that cite those sources.
3. The **verifier** reads each claim next to the text of the source it cites and gives a verdict: supported,
   unsupported or contradicted, with a word-for-word quote. Code then checks that the quote really is in the source.
4. A **coverage check** (plain code) asks whether each sub-question has at least 2 verified claims from at least
   2 different websites. If not, that sub-question goes back to research, up to 2 more times.
5. The **writer** produces the report from verified claims only. Any sentence that does not cite a verified claim
   is deleted, and citations are added by code, never by the model.

The result is a Markdown report with a summary, one section per sub-question, a **"Where sources disagree"**
section, a **"Not enough evidence"** section for anything still thin, the verification counts, and the list of
cited sources.

**Sharing:** the web page offers the report as a self-contained **HTML** file, a clean page that opens in any
browser and can be sent by email or chat, and saved as a PDF with *Print → Save as PDF*. A Markdown download is
also available.

## How the agents cooperate

```
question
  -> Planner      splits the question into up to 4 sub-questions
  -> Researchers  search the web and write claims, each citing a source (one per sub-question, in parallel)
  -> Verifier     checks every claim against its cited source
  -> Coverage     enough verified claims? If not, research that sub-question again (up to 2 more times)
  -> Writer       writes the report from verified claims only
```

The coordinator is [deepdive/graph.py](deepdive/graph.py), a LangGraph with one node per step. The loop from
Coverage back to Researchers is a real graph edge. The steps pass their results to each other through one shared
state: the sub-questions, sources, claims with their verdicts, and conflicts.

| Agent | Job | Tools | Output | Model (via Mesh) |
|---|---|---|---|---|
| Planner | Split the question into focused, searchable sub-questions | none | `Plan` | `openai/gpt-4o-mini` |
| Researcher | Decide what to search, then draft claims that each cite one source | `web_search` (Tavily, max 3 per attempt) | `Findings` | `openai/gpt-4o-mini` |
| Verifier | Judge each claim against only its cited source; list where sources disagree | none, by design | `Verification` | `google/gemini-2.5-flash` |
| Writer | Write the report from verified claims, citing claim ids per sentence | none | `Draft` | `openai/gpt-4o-mini` |

Each agent has its own prompt, its own output schema and its own function in
[deepdive/agents.py](deepdive/agents.py). All of them are built and run by one helper, `run_agent()` in
[deepdive/llm.py](deepdive/llm.py).

## How the report stays true to its sources

The language models propose; plain code checks. A claim reaches the report only if it passes every check:

1. **It cites a source the researcher actually retrieved.** Claims citing an unknown source id are dropped.
2. **The verifier says the source supports it.** The verifier sees only the cited source text, not the web.
3. **The verifier's quote is really in the source** ([deepdive/rules.py](deepdive/rules.py)). Every part of the
   quote (4+ words) must appear word for word; otherwise the claim is rejected, so the verifier cannot invent
   support either.
4. **Every report sentence cites verified claims only** ([deepdive/report.py](deepdive/report.py)). Sentences
   citing nothing, or citing a rejected claim, are deleted before the report is built.

An example from a real run, rejected and kept out of the report:

> **REJECTED C24 (unsupported):** *"General Motors and Samsung are building a battery cell plant in Indiana, with
> an investment of USD 3 billion, as part of GM's strategy to expand its EV portfolio."*
> **Reason:** The provided source discusses Ford's lithium partnerships and does not contain any information about
> General Motors, Samsung, or a battery cell plant in Indiana.

## Audit trail

Every run is stored in SQLite (`data/deepdive.db`, [deepdive/db.py](deepdive/db.py)):

- **`runs`**: the question, status, final report, and the final state: every sub-question, source, and claim with
  its verdict, quote and reason.
- **`log`**: a timestamped line for each step: the sub-questions, every search, `PASSED` / `REJECTED` for every
  claim with the reason, loop-backs, conflicts and dropped sentences.

The web page shows this in the **Claims checked** and **Log** tabs; the API returns it from `GET /runs/{id}`.

## Running it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env              # add MESH_API_KEY (rsk_...) and TAVILY_API_KEY (tvly-...)
```

| What | Command |
|---|---|
| Web page | `uv run streamlit run ui.py` → http://localhost:8501 |
| REST API (FastAPI) | `uv run uvicorn deepdive.api:app` → http://localhost:8000/docs |
| Tests (fake models, no network, no keys) | `uv run pytest` |
| Self-evaluation | `uv run python -m eval.run_eval` → writes `eval/results.md` |
| Try one agent on its own | `uv run python -m test.try_planner "your question"` (also `try_researcher`, `try_full_run`) |

API endpoints: `POST /research` runs a question and returns the report; `GET /runs` lists recent runs;
`GET /runs/{id}` returns one run with its claims, verdicts and full log.

Settings such as models, search limits and coverage thresholds are in [deepdive/config.py](deepdive/config.py).
The model ids can be overridden in `.env`.

## Key design decisions

- **Four distinct agents, one coordinator.** Only the researcher has a tool, because it is the only agent that
  needs to decide what to look up. The verifier deliberately has none: it may judge only the cited text.
- **Three model families.** The researcher and writer use OpenAI, the verifier Google Gemini, and the evaluation
  judge Anthropic Claude, so no model grades its own work. The cheap `gpt-4o-mini` does most of the calls.
- **Checks in code, not in prompts.** Source ids, quote matching, coverage and citation filtering are plain Python,
  so a model mistake cannot slip through unnoticed.
- **Coverage is a simple, visible rule:** at least 2 verified claims from at least 2 websites. Loop-backs are
  predictable, explainable and capped at 2 retries, and a researcher that goes back is told what was missing.
- **Fail closed.** If the verifier fails, its claims count as unverified and stay out of the report. If the daily
  model quota runs out, the run stops at once instead of producing a partial report.
- **Parallel where it is safe.** Sub-questions are independent, so they are researched and verified at the same
  time: about 1 to 2 minutes per question instead of 2 or more.
- **One web page, one source.** If two researchers find the same page, it keeps one source id and its text is
  merged, so the source list has no duplicates.
- **Robust to model quirks seen in real runs.** Agents return native JSON (`ProviderStrategy`), because tool-based
  output made `gpt-4o-mini` split its answer. The quote check accepts quotes joined from separate parts of a
  source (as long as each part is verbatim). Writer sentences that put claim ids inside the text are repaired.
  Rate-limited calls wait and retry.
- **Deliberately left out:** source-trust weighting, uploaded documents and conversational follow-up, to keep the
  system small and easy to follow.

## Self-evaluation

[eval/questions.json](eval/questions.json) has 12 test questions: factual, analytical, contradictory (sources are
known to disagree), one deliberately thin on evidence, and one with a known answer.
[eval/run_eval.py](eval/run_eval.py) runs each through the full pipeline and measures:

| Metric | Meaning |
|---|---|
| Verification pass rate | Claims that passed verification ÷ all claims the researchers drafted |
| Grounded sentences | Report sentences that an **independent judge** (Claude Haiku 4.5) finds supported by their cited sources |
| Sub-question coverage | Sub-questions that ended well supported |
| Extra research rounds | How often the loop-back was used |
| Known-answer fact recall | Expected facts found in the report (known-answer question only) |

### Results (2026-10-29, 12 of 12 questions completed)

Models via Mesh: researcher and writer `openai/gpt-4o-mini`, verifier `google/gemini-2.5-flash`,
judge `anthropic/claude-haiku-4.5`. Web search: Tavily (advanced).

| Metric | Average |
|---|---|
| **Grounded sentences (independent judge)** | **0.97** |
| Verification pass rate | 0.82 |
| Sub-question coverage | 1.00 |
| Extra research rounds per question | 0.25 |
| Known-answer fact recall | 1.00 |
| Seconds per question | 119 |

| Question | Category | Claims | Pass rate | Grounded | Coverage | Retries | Fact recall |
|---|---|---|---|---|---|---|---|
| ev-risks | analytical | 24 | 0.92 | 0.96 | 1.00 | 0 | - |
| battery-share | factual | 30 | 0.90 | 1.00 | 1.00 | 1 | - |
| sodium-ion | analytical | 30 | 0.83 | 0.96 | 1.00 | 1 | - |
| pli-acc | factual | 24 | 0.83 | 1.00 | 1.00 | 0 | - |
| lithium-price | factual | 24 | 0.75 | 1.00 | 1.00 | 0 | - |
| solid-state | contradictory | 24 | 0.88 | 1.00 | 1.00 | 0 | - |
| eu-ai-act | factual | 24 | 0.67 | 0.94 | 1.00 | 0 | - |
| glp1 | analytical | 24 | 1.00 | 0.95 | 1.00 | 0 | - |
| remote-work | contradictory | 24 | 0.83 | 0.95 | 1.00 | 0 | - |
| small-modular | analytical | 24 | 0.83 | 1.00 | 1.00 | 0 | - |
| niche-thin | thin-evidence | 24 | 0.83 | 0.93 | 1.00 | 0 | - |
| known-fact | known-answer | 29 | 0.59 | 1.00 | 1.00 | 1 | 1.00 |

Full results, including the run id of every question (open it in the web page or `GET /runs/{id}` to see the full
log): [eval/results.md](eval/results.md) and `eval/results.json`.

### What the numbers show

- **97% of report sentences are supported by their cited sources**, according to a judge model from a different
  family than both the writer and the verifier. This is the main trust number. The remaining 3% are sentences the
  judge found not fully supported, so the report is not perfect, and we report that rather than hide it.
- **The verifier really filters:** 18% of drafted claims were rejected and left out of the reports (55 of 305). It
  catches wrong numbers in particular. In the known-answer run it rejected *"JWST is approximately 1,423,281
  kilometers from Earth"* because the source says 1,420,514 km, and *"one circuit every 168 days"* because the source
  says 6 months.
- **Every sub-question ended well supported** (coverage 1.00), and the known-answer question found both expected
  facts (launch on 25 December 2021, position at L2).
- **The loop-back was needed less than in our first evaluation** (0.25 extra rounds per question, against 1.17
  before). Fixes made between the two runs (accepting quotes joined from two parts of a source, and repairing
  writer and verifier output quirks) stopped good claims from being wrongly rejected, so fewer sub-questions looked
  thin. Our first evaluation scored 0.70 pass rate, 0.94 coverage and 0.97 grounded.
- **Limitation: coverage checks quantity, not relevance.** The deliberately thin question (`niche-thin`, revenue of
  battery recycling startups in Tamil Nadu in 2025) was marked fully covered. Its claims are true to their sources,
  but most are India-wide battery-recycling market sizes rather than the revenue of Tamil Nadu startups that was
  asked for. A check that verified claims actually answer their sub-question would let the report say "not enough
  evidence" here. That is the next improvement.

## Project layout

```
deepdive/
  config.py   settings: Mesh URL and key, models, search limits, coverage thresholds
  llm.py      run_agent(): builds an agent with create_agent and runs it through Mesh (with retries)
  search.py   web search (Tavily)
  agents.py   the four agents: prompt + output schema + function each
  graph.py    the LangGraph coordinator: steps, parallel research/verify, loop-back, run_research()
  rules.py    plain-code checks: quote is in the source, enough proof per sub-question
  report.py   final report: drops unverified sentences, adds citations, conflicts, gaps and sources
  db.py       SQLite audit trail (runs + step-by-step log)
  api.py      FastAPI endpoints
ui.py         Streamlit web page
eval/         test questions, evaluation script, results
test/         tests (fake models, no network) and try_*.py scripts to run one agent at a time.
```
