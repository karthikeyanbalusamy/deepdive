from deepdive import config
from deepdive.agents import Draft, Section, Sentence, Verification
from deepdive.report import render, to_html
from deepdive.rules import coverage_gap, quote_in_source

SOURCE = "CATL held 37.9% of the global EV battery market in 2024, according to SNE Research."


def test_quote_must_really_appear_in_the_source():
    assert quote_in_source("catl held 37.9%  of the global EV battery market", SOURCE)
    assert quote_in_source("“CATL held 37.9% of the global EV battery market”", SOURCE)
    assert not quote_in_source("CATL held 45% of the global EV battery market", SOURCE)
    assert not quote_in_source("CATL", SOURCE)  # too short to prove anything


def test_quote_may_join_parts_from_different_places_but_each_part_must_be_real():
    source = "CATL held 37.9% of the global EV battery market in 2024. Demand was strong. BYD ranked second with 17.2%."
    assert quote_in_source("CATL held 37.9% of the global EV battery market ... BYD ranked second with 17.2%", source)
    assert quote_in_source("CATL held 37.9% of the global EV battery market [...] BYD ranked second with 17.2%", source)
    assert quote_in_source("CATL held 37.9% of the global EV battery market in 2024. BYD ranked second with 17.2%.", source)
    # one invented part is enough to fail
    assert not quote_in_source("CATL held 37.9% of the global EV battery market ... BYD ranked second with 25%", source)


def claim(cid, source_id, verdict):
    return {"id": cid, "sub_q": "Q1", "text": f"claim {cid}", "source_id": source_id, "verdict": verdict, "reason": ""}


SOURCES = {
    "S1": {"id": "S1", "url": "https://www.reuters.com/a", "title": "Reuters"},
    "S2": {"id": "S2", "url": "https://iea.org/b", "title": "IEA"},
    "S3": {"id": "S3", "url": "https://reuters.com/c", "title": "Reuters again"},
}


def test_sub_question_needs_enough_verified_claims_from_different_websites():
    assert coverage_gap([claim("C1", "S1", "supported"), claim("C2", "S2", "supported")], SOURCES) == ""
    assert "website" in coverage_gap([claim("C1", "S1", "supported"), claim("C2", "S3", "supported")], SOURCES)
    assert "verified claim" in coverage_gap([claim("C1", "S1", "supported"), claim("C2", "S2", "unsupported")], SOURCES)
    assert config.MIN_VERIFIED_CLAIMS == 2


def test_claim_ids_written_inside_the_text_are_moved_to_claim_ids():
    sentence = Sentence.model_validate({"text": "Sodium is cheap and abundant (C2, C5)."})
    assert sentence.claim_ids == ["C2", "C5"]
    assert sentence.text == "Sodium is cheap and abundant."
    # ids given properly are kept as they are
    assert Sentence(text="Kept.", claim_ids=["C1"]).claim_ids == ["C1"]


def test_verifier_answer_with_null_conflicts_is_accepted():
    answer = Verification.model_validate({"checks": [], "conflicts": None})
    assert answer.conflicts == []


def test_report_keeps_only_sentences_backed_by_verified_claims():
    state = {
        "question": "Who leads?",
        "claims": [claim("C1", "S1", "supported"), claim("C2", "S2", "supported"), claim("C3", "S2", "unsupported")],
        "sources": list(SOURCES.values()),
        "sub_questions": [{"id": "Q2", "text": "Policy?", "status": "thin", "gap": "only 0 verified claim(s), need 2"}],
        "conflicts": ["S1 says 37.9%, S2 says 36%"],
    }
    draft = Draft(title="EV batteries", summary=[Sentence(text="CATL leads.", claim_ids=["C1", "C2"])], sections=[
        Section(heading="Leaders", sentences=[
            Sentence(text="Kept.", claim_ids=["C2"]),
            Sentence(text="Uses a rejected claim.", claim_ids=["C3"]),
            Sentence(text="No citation at all.", claim_ids=[]),
        ]),
    ])
    markdown, kept, dropped = render(state, draft)

    assert [s["text"] for s in kept] == ["CATL leads.", "Kept."]
    assert dropped == ["Uses a rejected claim.", "No citation at all."]
    assert "CATL leads. [S1][S2]" in markdown
    assert "Where sources disagree" in markdown and "Not enough evidence" in markdown
    assert "2 of 3 claims passed verification" in markdown


def test_html_export_is_a_complete_page_with_clickable_sources():
    page = to_html("# EV <batteries>\n\n- CATL leads. [S1]\n\n## Sources\n\n- [S1] [Reuters](https://reuters.com/a)\n",
                   title="Who leads <EV>?")
    assert page.startswith("<!doctype html>")
    assert "<title>Who leads &lt;EV&gt;?</title>" in page  # the question is escaped, so it cannot break the page
    assert "<h2>Sources</h2>" in page
    assert '<a href="https://reuters.com/a">Reuters</a>' in page
