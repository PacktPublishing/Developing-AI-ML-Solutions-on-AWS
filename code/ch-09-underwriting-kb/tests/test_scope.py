"""What a search may read, how the hybrid query is built, and who is asking."""

import base64
import json

from audit import underwriter
from opensearch_store import query_body, scope


def _flat(where: dict) -> str:
    """Render a filter as JSON text, to check what it contains."""
    return json.dumps(where, sort_keys=True)


def test_policy_stays_in_scope_whatever_the_filters():
    """A sector filter narrows the precedent, never the policy that applies to it."""
    where = scope({"sector": "Logistics", "max_exposure": 2_000_000})
    should = where["bool"]["should"]
    assert {"term": {"source": "policy"}} in should
    cases = should[0]["bool"]["filter"]
    assert {"terms": {"source": ["corpus"]}} in cases
    assert {"term": {"sector": "Logistics"}} in cases
    assert {"range": {"exposure_usd": {"lte": 2_000_000}}} in cases


def test_a_document_under_review_is_never_in_scope():
    """The default scope reads decided cases and policy, not submissions."""
    assert "submission" not in _flat(scope())


def test_blank_filters_add_nothing():
    """An empty form adds no clause beyond the source."""
    where = scope({"sector": "", "min_exposure": None, "since": None})
    assert where["bool"]["should"][0]["bool"]["filter"] == [
        {"terms": {"source": ["corpus"]}}
    ]


def test_hybrid_runs_keyword_and_vector_under_the_same_scope():
    """Both halves of a hybrid query carry the filter, and the keyword half reads the title."""
    where = scope({"sector": "Pharmacy"})
    body = query_body([0.1, 0.2], "loan 99000098", 5, where, hybrid=True)
    keyword, knn = body["query"]["hybrid"]["queries"]
    assert keyword["bool"]["filter"] == where
    assert "title^2" in keyword["bool"]["must"]["multi_match"]["fields"]
    assert knn["knn"]["embedding"]["filter"] == where
    plain = query_body([0.1, 0.2], "loan 99000098", 5, where, hybrid=False)
    assert "hybrid" not in plain["query"]


def test_underwriter_comes_from_the_token_claims():
    """The audit line names the signed-in underwriter, and "local" without a token."""
    claims = (
        base64.urlsafe_b64encode(
            json.dumps({"cognito:username": "underwriter1", "sub": "abc"}).encode()
        )
        .decode()
        .rstrip("=")
    )
    assert underwriter(f"Bearer header.{claims}.signature") == "underwriter1"
    assert underwriter(None) == "local"
