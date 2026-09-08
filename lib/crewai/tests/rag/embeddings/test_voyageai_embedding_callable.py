"""Tests for the VoyageAI by MongoDB embedding function.

The real ``voyageai`` package is optional and not installed in CI, so a mock
module is injected into ``sys.modules`` before importing the callable. This lets
the lazy ``import voyageai`` inside the embedding function resolve to the mock.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest


@contextmanager
def mock_voyageai_client():
    """Inject a mock ``voyageai`` module and yield the mock client instance."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Client.return_value = mock_client
    original = sys.modules.get("voyageai")
    sys.modules["voyageai"] = mock_module
    try:
        yield mock_client
    finally:
        if original is not None:
            sys.modules["voyageai"] = original
        else:
            sys.modules.pop("voyageai", None)


def _make_function(mock_client, **config):
    from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
        VoyageAIEmbeddingFunction,
    )

    config.setdefault("api_key", "test-key")
    return VoyageAIEmbeddingFunction(**config)


def test_plain_model_uses_embed():
    """Non-contextualized models route through client.embed with a flat list."""
    with mock_voyageai_client() as client:
        client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        fn = _make_function(client, model="voyage-3.5")

        result = fn(["hello", "world"])

        np.testing.assert_allclose(result, [[0.1, 0.2], [0.3, 0.4]])
        client.embed.assert_called_once()
        assert client.embed.call_args.kwargs["texts"] == ["hello", "world"]
        assert client.embed.call_args.kwargs["model"] == "voyage-3.5"
        client.contextualized_embed.assert_not_called()


def test_string_input_wrapped_in_list():
    """A single string input is wrapped into a list before embedding."""
    with mock_voyageai_client() as client:
        client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2]])
        fn = _make_function(client, model="voyage-3.5")

        fn("hello")

        assert client.embed.call_args.kwargs["texts"] == ["hello"]


def test_contextualized_document_path():
    """voyage-context models send a flat list with auto-chunking + chunk_size."""
    with mock_voyageai_client() as client:
        client.contextualized_embed.return_value = MagicMock(
            results=[
                MagicMock(embeddings=[[0.1, 0.2]]),
                MagicMock(embeddings=[[0.3, 0.4]]),
            ]
        )
        fn = _make_function(client, model="voyage-context-4")

        result = fn(["doc one", "doc two"])

        np.testing.assert_allclose(result, [[0.1, 0.2], [0.3, 0.4]])
        client.embed.assert_not_called()
        kwargs = client.contextualized_embed.call_args.kwargs
        assert kwargs["inputs"] == ["doc one", "doc two"]
        assert kwargs["model"] == "voyage-context-4"
        assert kwargs["input_type"] == "document"
        assert kwargs["enable_auto_chunking"] is True
        assert kwargs["chunk_size"] == 32000


def test_contextualized_query_path_disables_auto_chunking():
    """Query embeddings disable auto-chunking and omit chunk_size."""
    with mock_voyageai_client() as client:
        client.contextualized_embed.return_value = MagicMock(
            results=[MagicMock(embeddings=[[0.5, 0.6]])]
        )
        fn = _make_function(client, model="voyage-context-4", input_type="query")

        result = fn(["a query"])

        np.testing.assert_allclose(result, [[0.5, 0.6]])
        kwargs = client.contextualized_embed.call_args.kwargs
        assert kwargs["input_type"] == "query"
        assert kwargs["enable_auto_chunking"] is False
        assert "chunk_size" not in kwargs


def test_contextualized_one_vector_per_input():
    """Each input resolves to exactly one vector (first chunk)."""
    with mock_voyageai_client() as client:
        client.contextualized_embed.return_value = MagicMock(
            results=[MagicMock(embeddings=[[1.0], [9.9]])]
        )
        fn = _make_function(client, model="voyage-context-3")

        result = fn(["only doc"])

        np.testing.assert_allclose(result, [[1.0]])


def test_missing_voyageai_raises_import_error():
    """A helpful ImportError is raised when voyageai is not installed."""
    original = sys.modules.pop("voyageai", None)
    sys.modules["voyageai"] = None  # force import to fail
    try:
        from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
            VoyageAIEmbeddingFunction,
        )

        with pytest.raises(ImportError, match="voyageai is required"):
            VoyageAIEmbeddingFunction(api_key="k")
    finally:
        if original is not None:
            sys.modules["voyageai"] = original
        else:
            sys.modules.pop("voyageai", None)
