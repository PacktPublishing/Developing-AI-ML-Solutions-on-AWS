"""Test submission versioning and document isolation with an in-memory store."""

import opensearch_store
import pytest
from opensearch_store import OpenSearchStore


class FakeIndices:
    """The indices half of the client: the store asks it three things."""

    def __init__(self, client) -> None:
        """Hold the client whose documents these indices describe."""
        self.client = client

    def exists(self, index: str) -> bool:
        """Report that the index exists, as it does once anything is written."""
        return True

    def refresh(self, index: str) -> None:
        """Nothing is buffered, so a refresh has nothing to do."""

    def delete(self, index: str) -> None:
        """Drop every document."""
        self.client.docs.clear()


class FakeClient:
    """An in-memory stand-in for the subset of OpenSearch the store uses."""

    def __init__(self) -> None:
        """Start with no documents."""
        self.docs: dict[str, dict] = {}
        self.indices = FakeIndices(self)
        self._auto = 0

    # the store writes through helpers.bulk, which is patched to call this
    def bulk(self, actions) -> None:
        """Index each action, replacing a document with the same id."""
        for action in actions:
            doc = {k: v for k, v in action.items() if not k.startswith("_")}
            key = action.get("_id")
            if not key:
                self._auto += 1
                key = f"auto-{self._auto}"
            self.docs[key] = doc

    @staticmethod
    def _matches(doc: dict, filters: list[dict]) -> bool:
        """Report whether one document satisfies every term filter."""
        for clause in filters:
            field, wanted = next(iter(clause["term"].items()))
            if doc.get(field) != wanted:
                return False
        return True

    def _selected(self, body: dict) -> list[dict]:
        """Return the documents a query body selects."""
        filters = body["query"]["bool"]["filter"]
        return [d for d in self.docs.values() if self._matches(d, filters)]

    def search(self, index: str, body: dict, params: dict | None = None) -> dict:
        """Answer the aggregation and the sorted search the store sends."""
        selected = self._selected(body)
        if "aggs" in body:
            versions = [d.get("submission_version") or 0 for d in selected]
            return {"aggregations": {"latest": {"value": max(versions, default=None)}}}
        for key in reversed(body.get("sort", [])):
            field, order = next(iter(key.items()))
            selected.sort(key=lambda d, f=field: d.get(f) or 0, reverse=order == "desc")
        return {"hits": {"hits": [{"_source": d, "_score": 1.0} for d in selected]}}

    def delete_by_query(self, index: str, body: dict, refresh: bool = False) -> dict:
        """Delete what the query selects and report how many went."""
        doomed = [
            k
            for k, d in self.docs.items()
            if self._matches(d, body["query"]["bool"]["filter"])
        ]
        for key in doomed:
            del self.docs[key]
        return {"deleted": len(doomed)}


@pytest.fixture
def store(monkeypatch):
    """Return an OpenSearchStore on the fake client, with a stub embedder."""
    client = FakeClient()
    monkeypatch.setattr(opensearch_store, "client", lambda: client)
    monkeypatch.setattr(
        opensearch_store, "embed", lambda _r, texts: [[0.0]] * len(texts)
    )

    class FakeHelpers:
        """Stand in for opensearchpy.helpers.bulk."""

        @staticmethod
        def bulk(client, actions, **kwargs):
            """Hand the actions to the fake client."""
            client.bulk(actions)

    monkeypatch.setattr(opensearch_store, "helpers", FakeHelpers)
    return OpenSearchStore()


# ---------------------------------------------------------------------------
# A second upload is the one that answers
# ---------------------------------------------------------------------------
def test_a_corrected_statement_replaces_the_figures_of_the_first(store):
    """Read the latest uploaded statement figures by default."""
    store.index_submission(
        None, "CASE-1", "CASE-1", ["first"], meta={"monthly_income": 50_000}
    )
    store.index_submission(
        None, "CASE-1", "CASE-1", ["second"], meta={"monthly_income": 30_000}
    )
    doc = store.read_submission("CASE-1")
    assert doc["version"] == 2
    assert doc["monthly_income"] == 30_000
    assert doc["text"] == "second"


def test_the_earlier_version_is_still_there_to_be_asked_for(store):
    """Allow retrieval of an earlier statement version."""
    store.index_submission(
        None, "CASE-1", "CASE-1", ["first"], meta={"monthly_income": 50_000}
    )
    store.index_submission(
        None, "CASE-1", "CASE-1", ["second"], meta={"monthly_income": 30_000}
    )
    assert store.read_submission("CASE-1", version=1)["monthly_income"] == 50_000


def test_a_shorter_correction_does_not_leave_the_longer_one_behind(store):
    """Read only chunks belonging to the selected statement version."""
    store.index_submission(
        None, "CASE-1", "CASE-1", ["head", "march", "april"], meta={"months": 3}
    )
    store.index_submission(
        None, "CASE-1", "CASE-1", ["head", "may"], meta={"months": 1}
    )
    doc = store.read_submission("CASE-1")
    assert doc["months"] == 1
    assert "march" not in doc["text"] and "april" not in doc["text"]


def test_indexing_the_same_version_twice_is_not_two_copies(store):
    """A retry after a timeout writes the same ids, so it overwrites itself."""
    store.index_submission(None, "CASE-1", "CASE-1", ["one", "two"])
    before = len(store.client.docs)
    store.client.docs.clear()
    store.index_submission(None, "CASE-1", "CASE-1", ["one", "two"])
    store.index_submission(None, "CASE-1", "CASE-1", ["one", "two"])
    assert before == 2
    # two versions of two chunks, not four copies of one
    assert len(store.client.docs) == 4


def test_a_case_with_nothing_uploaded_reads_empty(store):
    """Return an empty result for a case without a statement."""
    assert store.read_submission("CASE-NONE") == {}
    assert store.latest_version("CASE-NONE") == 0


def test_dropping_a_case_takes_every_version_with_it(store):
    """Delete all statement versions when no version is specified."""
    store.index_submission(None, "CASE-1", "CASE-1", ["first"])
    store.index_submission(None, "CASE-1", "CASE-1", ["second"])
    assert store.drop_submission("CASE-1") == 2
    assert store.read_submission("CASE-1") == {}


# ---------------------------------------------------------------------------
# Supporting documents accumulate rather than replace
# ---------------------------------------------------------------------------
def test_supporting_documents_do_not_replace_each_other(store):
    """Return all uploaded supporting documents."""
    store.index_submission(
        None,
        "CASE-1",
        "tax filing",
        ["tax body"],
        meta={"doc_type": "tax filing"},
        source="context",
        doc_name="tax filing",
    )
    store.index_submission(
        None,
        "CASE-1",
        "supply contract",
        ["contract body"],
        meta={"doc_type": "contract"},
        source="context",
        doc_name="supply contract",
    )
    documents = store.list_submissions("CASE-1", source="context")
    assert [d["doc_name"] for d in documents] == ["tax filing", "supply contract"]
    assert [d["version"] for d in documents] == [1, 2]


def test_a_supporting_document_is_not_the_statement(store):
    """Keep statement and supporting-document queries separate."""
    store.index_submission(None, "CASE-1", "CASE-1", ["statement"])
    store.index_submission(
        None, "CASE-1", "invoice", ["invoice"], source="context", doc_name="invoice"
    )
    assert store.read_submission("CASE-1")["text"] == "statement"
    assert store.list_submissions("CASE-1", source="context")[0]["text"] == "invoice"
    assert (
        store.list_submissions("CASE-1", source="submission")[0]["text"] == "statement"
    )
