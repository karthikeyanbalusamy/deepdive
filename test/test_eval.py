from eval.run_eval import fact_recall

REPORT = "JWST was launched on 25 December 2021 and sits at the second Lagrange point (L2)."


def test_fact_recall_accepts_any_listed_spelling():
    assert fact_recall(REPORT, [["December 25, 2021", "25 December 2021"], "L2"]) == 1.0
    assert fact_recall(REPORT, ["December 25, 2021", "L2"]) == 0.5  # only one spelling allowed: date missed
    assert fact_recall(REPORT, ["L1"]) == 0.0
