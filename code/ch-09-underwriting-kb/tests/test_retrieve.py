"""Every source an answer drew on is cited, policies included."""

from retrieve import _sources_list


def test_distinct_policies_are_each_cited():
    """Policies share loan id 0, so each must still be its own source."""
    hits = [
        (1042, "Harbor Freight Ltd", "...", 0.9),
        (0, "POL-DTI", "...", 0.8),
        (1042, "Harbor Freight Ltd", "...", 0.7),
        (0, "POL-SECURITY", "...", 0.6),
        (0, "POL-DTI", "...", 0.5),
    ]
    assert _sources_list(hits) == [
        {"loan_id": 1042, "borrower": "Harbor Freight Ltd"},
        {"loan_id": 0, "borrower": "POL-DTI"},
        {"loan_id": 0, "borrower": "POL-SECURITY"},
    ]
