"""Web search with Tavily. Every result keeps its URL, so claims can be traced back to their source."""

from tavily import TavilyClient

from deepdive import config
from deepdive.llm import QuotaExceeded


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Return results as {url, title, text}, where text is the part of the page relevant to the query."""
    if not config.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
    try:
        response = TavilyClient(api_key=config.TAVILY_API_KEY).search(query, search_depth="advanced", max_results=max_results)
    except Exception as exc:
        if "usage limit" in str(exc).lower():  # out of search credits: every later search would fail too
            raise QuotaExceeded(f"Web search limit reached (Tavily). Add credits or wait for the reset. ({exc})") from exc
        raise
    return [
        {"url": r["url"], "title": r.get("title") or r["url"], "text": r["content"]}
        for r in response["results"]
        if r.get("content")
    ]
