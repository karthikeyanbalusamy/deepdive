"""REST API.  Run with:  uv run uvicorn deepdive.api:app --port 8000   (docs at /docs)

    POST /research     run a research question, returns the report
    GET  /runs         recent runs
    GET  /runs/{id}    one run: report, claims with verdicts, and the full log
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deepdive import db
from deepdive.graph import run_research

app = FastAPI(title="DeepDive", description="Self-verifying research analyst")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500)


@app.post("/research")
def research(request: ResearchRequest) -> dict:
    state = run_research(request.question.strip())
    return {"run_id": state["run_id"], "report": state["report"]}


@app.get("/runs")
def list_runs() -> list[dict]:
    return db.list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run
