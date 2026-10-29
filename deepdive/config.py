"""All settings in one place. Keys and models are read from the .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Every LLM call goes through Mesh API (mandatory for the hackathon).
MESH_BASE_URL = "https://api.meshapi.ai/v1"
MESH_API_KEY = os.getenv("MESH_API_KEY", "")

# Web search.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Mesh model ids. Three different model families, so claims are never checked by the model family that wrote them.
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")                          # planner, researcher, writer
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "google/gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-haiku-4.5")      # used only by the evaluation

# Research limits.
MAX_SUB_QUESTIONS = 4
MAX_SEARCHES = 3      # web searches per researcher, per attempt
MAX_RETRIES = 2       # extra research attempts for a thinly supported sub-question

# A sub-question is "well supported" when it has this many verified claims from this many websites.
MIN_VERIFIED_CLAIMS = 2
MIN_WEBSITES = 2

DB_PATH = Path(os.getenv("DB_PATH", "data/deepdive.db"))
