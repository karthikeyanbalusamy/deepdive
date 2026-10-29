"""LLM access. Every agent is built here with LangChain's create_agent, and every model call
goes through Mesh API (an OpenAI-compatible gateway)."""

import logging
import time

import openai
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from deepdive import config

log = logging.getLogger(__name__)

RATE_LIMIT_WAITS = [20, 40]  # seconds to wait before each retry when the provider says "too many requests"


class QuotaExceeded(RuntimeError):
    """The provider's daily request limit is used up. Retrying will not help, so the run must stop."""


def chat_model(model: str) -> ChatOpenAI:
    if not config.MESH_API_KEY:
        raise RuntimeError("MESH_API_KEY is not set. Add it to your .env file.")
    return ChatOpenAI(
        model=model,
        base_url=config.MESH_BASE_URL,
        api_key=config.MESH_API_KEY,
        temperature=0,
        timeout=120,
        max_retries=2,
    )


def run_agent(model: str, prompt: str, schema: type[BaseModel], message: str, tools: list | None = None):
    """Run one agent on one message and return its answer as an instance of `schema`.

    Example: run_agent(model, prompt, Plan, question) returns a Plan object.
    """
    agent = create_agent(
        model=chat_model(model),
        tools=tools or [],
        system_prompt=prompt,
        # The model replies directly in the JSON format of `schema`. (Returning the answer through a
        # tool call instead makes some models, e.g. gpt-4o-mini, split it into several answers.)
        response_format=ProviderStrategy(schema),
    )
    result = _invoke_with_retry(agent, message)
    answer = result.get("structured_response")
    if answer is None:
        raise RuntimeError(f"The model did not return a {schema.__name__}.")
    return answer


def _invoke_with_retry(agent, message: str) -> dict:
    """Run the agent; if the provider is rate-limiting us, wait and try again a couple of times."""
    for wait in RATE_LIMIT_WAITS + [None]:
        try:
            return agent.invoke({"messages": [{"role": "user", "content": message}]}, {"recursion_limit": 20})
        except openai.RateLimitError as exc:
            if "per-day" in str(exc):
                raise QuotaExceeded(f"Daily model limit reached. Wait for the reset or add credit. ({exc})") from exc
            if wait is None:
                raise
            log.warning("Rate limited, retrying in %ss", wait)
            time.sleep(wait)
