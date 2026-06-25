"""
Tests for the MARENGO (TwelveLabs) search mode.

Run with: python -W ignore -m pytest test_marengo.py

The live test is skipped unless TWELVELABS_API_KEY is set in the environment.
The ranking test runs everywhere as it stubs out the network call.
"""
import os

import pytest

from scoper import Scoper


def _make_scoper():
    # small_dataset trains a tiny Word2Vec on the Brown corpus; that is
    # unrelated to MARENGO but keeps the constructor happy.
    return Scoper(small_dataset=True)


def test_marengo_ranks_by_cosine_similarity(monkeypatch):
    """
    No-network unit test: stub the embedding call with fixed vectors and
    assert captions come back ordered by cosine similarity to the query.
    """
    finder = _make_scoper()

    vectors = {
        "apple watch": [1.0, 0.0, 0.0],
        "the new apple watch is great": [0.9, 0.1, 0.0],  # closest
        "talking about laptops": [0.0, 1.0, 0.0],          # orthogonal
        "completely unrelated text": [-1.0, 0.0, 0.0],      # opposite
    }
    monkeypatch.setattr(finder, "_get_marengo_embedding", lambda text: vectors[text])

    corpus = [
        "the new apple watch is great",
        "talking about laptops",
        "completely unrelated text",
    ]
    ranked = finder.get_captions_by_marengo_similarity("apple watch", corpus, limit=2)

    assert ranked[0] == "the new apple watch is great"
    assert "completely unrelated text" not in ranked


def test_marengo_empty_corpus(monkeypatch):
    finder = _make_scoper()
    monkeypatch.setattr(finder, "_get_marengo_embedding", lambda text: [1.0, 0.0])
    assert finder.get_captions_by_marengo_similarity("query", [], limit=5) == []


@pytest.mark.skipif(
    not os.environ.get("TWELVELABS_API_KEY"),
    reason="TWELVELABS_API_KEY not set; skipping live Marengo embedding test",
)
def test_marengo_live_embedding_dimension():
    """Live test: a real Marengo text embedding is a 512-dim vector."""
    finder = _make_scoper()
    vec = finder._get_marengo_embedding("apple watch")
    assert len(vec) == 512
