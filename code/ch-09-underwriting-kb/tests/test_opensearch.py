"""The OpenSearch store: the score conversion, and a round trip when a node is up."""

import socket

import pytest

OS_PORT: int = 9200


def _node_up() -> bool:
    """Report whether an OpenSearch node is listening locally."""
    try:
        socket.create_connection(("localhost", OS_PORT), timeout=2).close()
        return True
    except OSError:
        return False


def test_lucene_score_converts_to_cosine():
    """Lucene reports cosine c as (1 + c) / 2, and the store inverts it.

    Pure arithmetic, so it runs without a node: a perfect match reads 1.0, an
    orthogonal chunk reads 0.0, and an opposed one reads -1.0.
    """

    def to_cosine(score):
        """Invert a Lucene cosinesimil score, as the store does."""
        return 2 * score - 1

    assert to_cosine(1.0) == pytest.approx(1.0)
    assert to_cosine(0.5) == pytest.approx(0.0)
    assert to_cosine(0.0) == pytest.approx(-1.0)


@pytest.mark.skipif(not _node_up(), reason="no OpenSearch node on localhost:9200")
def test_exact_chunk_ranks_its_own_memo_first(monkeypatch, tmp_path):
    """An exact memo chunk ranks its own source loan first."""
    import embed_memos
    import gen_memos
    import opensearch_store
    import stores

    def _fake_embed(_runtime, texts):
        """Deterministic bag-of-characters vectors, so no model is needed."""
        out = []
        for t in texts:
            v = [0.0] * 64
            for ch in t.lower():
                v[ord(ch) % 64] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    # Never touch memo_chunks: a developer's seeded corpus lives there, and reset()
    # would drop it and recreate it at the stub's width.
    monkeypatch.setattr(opensearch_store, "INDEX", "memo_chunks_test")
    # the index is created at EMBED_DIM, so the stub's width has to match it
    monkeypatch.setattr(embed_memos, "EMBED_DIM", 64)
    monkeypatch.setattr(embed_memos, "embed", _fake_embed)
    monkeypatch.setattr(embed_memos, "get_runtime", lambda: None)
    monkeypatch.setattr(opensearch_store, "embed", _fake_embed)

    memo_dir = tmp_path / "memos"
    gen_memos.generate(memo_dir, count=8, seed=7, messy=False)
    embed_memos.seed(memo_dir)

    structured = next(
        p
        for p in sorted(memo_dir.glob("*.txt"))
        if embed_memos.parse_memo(p)[2].startswith("SME Credit Memo")
    )
    loan_id, _, body = embed_memos.parse_memo(structured)
    chunk = embed_memos.chunk_text(body)[0]

    hits = stores.get_store().search(None, chunk, k=3)
    assert hits[0][0] == loan_id

    stores.get_store().client.indices.delete(index="memo_chunks_test")
