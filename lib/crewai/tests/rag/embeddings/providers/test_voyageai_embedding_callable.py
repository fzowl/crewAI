"""Tests for the VoyageAI by MongoDB embedding function."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
    VoyageAIEmbeddingFunction,
    _is_contextualized_model,
)


@pytest.fixture
def fake_voyageai(monkeypatch):
    """Install a fake ``voyageai`` module exposing a mock Client."""
    client = MagicMock()
    module = SimpleNamespace(Client=MagicMock(return_value=client))
    monkeypatch.setitem(sys.modules, "voyageai", module)
    return client


@pytest.mark.parametrize(
    "model,expected",
    [
        ("voyage-context-4", True),
        ("voyage-context-3", True),
        ("voyage-3.5", False),
        ("voyage-code-4", False),
    ],
)
def test_is_contextualized_model(model, expected):
    assert _is_contextualized_model(model) is expected


def test_standard_model_uses_embed(fake_voyageai):
    fake_voyageai.embed.return_value = SimpleNamespace(
        embeddings=[[0.1, 0.2], [0.3, 0.4]]
    )

    fn = VoyageAIEmbeddingFunction(api_key="key", model="voyage-3.5")
    result = fn(["hello", "world"])

    np.testing.assert_allclose(np.asarray(result), [[0.1, 0.2], [0.3, 0.4]])
    fake_voyageai.embed.assert_called_once()
    fake_voyageai.contextualized_embed.assert_not_called()
    assert fake_voyageai.embed.call_args.kwargs["model"] == "voyage-3.5"


def test_contextualized_model_routes_to_contextualized_embed(fake_voyageai):
    fake_voyageai.contextualized_embed.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(embeddings=[[0.1, 0.2]]),
            SimpleNamespace(embeddings=[[0.3, 0.4]]),
        ]
    )

    fn = VoyageAIEmbeddingFunction(api_key="key", model="voyage-context-4")
    result = fn(["chunk one", "chunk two"])

    # One embedding per input document, unwrapped from its per-chunk group.
    np.testing.assert_allclose(np.asarray(result), [[0.1, 0.2], [0.3, 0.4]])
    fake_voyageai.embed.assert_not_called()
    call = fake_voyageai.contextualized_embed.call_args
    assert call.kwargs["model"] == "voyage-context-4"
    # ChromaDB documents are grouped one-per-list to scope contextualization.
    assert call.kwargs["inputs"] == [["chunk one"], ["chunk two"]]


def test_contextualized_embed_accepts_flat_list(fake_voyageai):
    """A flat ``list[str]`` is treated as a single document's chunks."""
    fake_voyageai.contextualized_embed.return_value = SimpleNamespace(results=[])

    fn = VoyageAIEmbeddingFunction(api_key="key", model="voyage-context-4")
    fn.contextualized_embed(["chunk a", "chunk b"])

    assert fake_voyageai.contextualized_embed.call_args.kwargs["inputs"] == [
        ["chunk a", "chunk b"]
    ]


def test_contextualized_embed_accepts_nested_list(fake_voyageai):
    """A ``list[list[str]]`` is forwarded unchanged."""
    fake_voyageai.contextualized_embed.return_value = SimpleNamespace(results=[])

    fn = VoyageAIEmbeddingFunction(api_key="key", model="voyage-context-4")
    fn.contextualized_embed([["doc1 chunk"], ["doc2 chunk a", "doc2 chunk b"]])

    assert fake_voyageai.contextualized_embed.call_args.kwargs["inputs"] == [
        ["doc1 chunk"],
        ["doc2 chunk a", "doc2 chunk b"],
    ]
