import httpx
import openai
import pytest

from deepdive import llm


def rate_limit_error(message: str) -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
    return openai.RateLimitError(message, response=response, body=None)


class FlakyAgent:
    """Fails with the given errors first, then answers."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, *args):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"structured_response": "answer"}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(llm, "RATE_LIMIT_WAITS", [0, 0])


def test_rate_limited_call_is_retried_until_it_succeeds():
    agent = FlakyAgent([rate_limit_error("too many requests"), rate_limit_error("too many requests")])
    assert llm._invoke_with_retry(agent, "hi") == {"structured_response": "answer"}
    assert agent.calls == 3


def test_gives_up_after_the_last_retry():
    agent = FlakyAgent([rate_limit_error("too many requests")] * 3)
    with pytest.raises(openai.RateLimitError):
        llm._invoke_with_retry(agent, "hi")
    assert agent.calls == 3


def test_daily_limit_is_not_retried():
    agent = FlakyAgent([rate_limit_error("Rate limit exceeded: free-models-per-day")])
    with pytest.raises(llm.QuotaExceeded):
        llm._invoke_with_retry(agent, "hi")
    assert agent.calls == 1


def test_tavily_usage_limit_becomes_quota_exceeded(monkeypatch):
    from deepdive import config, search

    class OutOfCreditsClient:
        def __init__(self, api_key):
            pass

        def search(self, *args, **kwargs):
            raise Exception("This request exceeds your plan's set usage limit. Please upgrade your plan.")

    monkeypatch.setattr(config, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(search, "TavilyClient", OutOfCreditsClient)
    with pytest.raises(llm.QuotaExceeded, match="Web search limit"):
        search.web_search("anything")
