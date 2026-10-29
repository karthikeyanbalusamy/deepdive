"""Plain-code checks. They do not trust any LLM."""

import re
from urllib.parse import urlparse

from deepdive import config


def normalize(text: str) -> str:
    """Lowercase, straighten quotes and collapse spaces, so small formatting differences don't matter."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip(" .\"'")


def quote_in_source(quote: str, source_text: str) -> bool:
    """True if the verifier's quote really appears in the source.

    A quote may join parts from different places in the source (e.g. "A ... B"). It is split into
    pieces at "...", "[...]", "…" and sentence ends, and every piece of 4+ words must appear in the
    source word for word. Shorter pieces are ignored, but at least one long piece is required.
    """
    pieces = re.split(r"\[\.\.\.\]|\.\.\.|…|(?<=[.!?])\s+", quote)
    pieces = [normalize(p) for p in pieces]
    pieces = [p for p in pieces if len(p.split()) >= 4]
    source = normalize(source_text)
    return bool(pieces) and all(p in source for p in pieces)


def website(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def coverage_gap(claims: list[dict], sources: dict[str, dict]) -> str:
    """Why a sub-question is thinly supported, or "" if it is well supported.

    `claims` are the claims of one sub-question.
    """
    verified = [c for c in claims if c["verdict"] == "supported"]
    websites = {website(sources[c["source_id"]]["url"]) for c in verified}
    problems = []
    if len(verified) < config.MIN_VERIFIED_CLAIMS:
        problems.append(f"only {len(verified)} verified claim(s), need {config.MIN_VERIFIED_CLAIMS}")
    if len(websites) < config.MIN_WEBSITES:
        problems.append(f"verified claims come from {len(websites)} website(s), need {config.MIN_WEBSITES}")
    return "; ".join(problems)
