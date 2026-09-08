"""Tests for the VoyageAI embedding function.

These tests mock the ``voyageai.Client`` so they run without network access or a
``VOYAGE_API_KEY``. They cover token-aware batching and the contextualized
document/query embedding paths.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("voyageai", reason="voyageai not installed")

from crewai.rag.embeddings.providers.voyageai.embedding_callable import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_TOTAL_TOKEN_LIMIT,
    VoyageAIEmbeddingFunction,
)


def _make_ef(model: str = "voyage-3.5", **config) -> VoyageAIEmbeddingFunction:
    """Build an embedding function with a fully mocked client."""
    with patch("voyageai.Client"):
        ef = VoyageAIEmbeddingFunction(api_key="test-key", model=model, **config)
    ef._client = MagicMock()
    return ef


def _mock_token_counts(ef: VoyageAIEmbeddingFunction, counts: list[int]) -> None:
    """Make ``client.tokenize`` report the given per-text token counts."""
    ef._client.tokenize.return_value = [["t"] * n for n in counts]


def _context_result(vectors: list[list[float]]) -> MagicMock:
    """Build a fake contextualized_embed result: one result per input."""
    result = MagicMock()
    result.results = []
    for vec in vectors:
        item = MagicMock()
        item.embeddings = [vec]  # exactly one chunk -> one embedding per input
        result.results.append(item)
    return result


def test_name() -> None:
    assert VoyageAIEmbeddingFunction.name() == "voyageai"


def test_model_type_detection() -> None:
    assert _make_ef("voyage-context-4")._is_context_model() is True
    assert _make_ef("voyage-3.5")._is_context_model() is False
    assert _make_ef("voyage-multimodal-3")._is_multimodal_model() is True
    assert _make_ef("voyage-3.5")._is_multimodal_model() is False


@pytest.mark.parametrize(
    "model,expected",
    [
        ("voyage-2", 320_000),
        ("voyage-3.5", 320_000),
        ("voyage-3.5-lite", 1_000_000),
        ("voyage-4", 320_000),
        ("voyage-4-lite", 1_000_000),
        ("voyage-4-large", 120_000),
        ("voyage-code-4", 120_000),
        ("voyage-context-4", 120_000),
        ("voyage-context-3", 120_000),
        ("voyage-multimodal-3", 32_000),
        ("some-unknown-model", DEFAULT_TOTAL_TOKEN_LIMIT),
    ],
)
def test_get_token_limit(model: str, expected: int) -> None:
    assert _make_ef(model).get_token_limit() == expected


# --- Token-aware batching --------------------------------------------------


def test_build_batches_splits_at_token_boundary() -> None:
    ef = _make_ef("voyage-3.5")
    _mock_token_counts(ef, [60, 30, 60])
    with patch.object(ef, "get_token_limit", return_value=100):
        batches = list(ef._build_batches(["a", "b", "c"]))
    # a(60)+b(30)=90 fit; adding c(60) would exceed 100 -> new batch.
    assert batches == [["a", "b"], ["c"]]


def test_build_batches_single_oversized_text_goes_alone() -> None:
    ef = _make_ef("voyage-3.5")
    _mock_token_counts(ef, [5_000])
    with patch.object(ef, "get_token_limit", return_value=100):
        batches = list(ef._build_batches(["huge"]))
    # A single text over the limit is still emitted, on its own.
    assert batches == [["huge"]]


def test_build_batches_oversized_text_isolated_between_others() -> None:
    ef = _make_ef("voyage-3.5")
    _mock_token_counts(ef, [50, 5_000, 50])
    with patch.object(ef, "get_token_limit", return_value=100):
        batches = list(ef._build_batches(["a", "big", "c"]))
    assert batches == [["a"], ["big"], ["c"]]


def test_build_batches_respects_item_count_cap() -> None:
    ef = _make_ef("voyage-3.5")
    n = BATCH_SIZE * 2 + 5
    texts = [f"t{i}" for i in range(n)]
    _mock_token_counts(ef, [1] * n)  # token cap never binds
    batches = list(ef._build_batches(texts))
    assert [len(b) for b in batches] == [BATCH_SIZE, BATCH_SIZE, 5]
    assert sum(len(b) for b in batches) == n


def test_build_batches_empty() -> None:
    ef = _make_ef("voyage-3.5")
    assert list(ef._build_batches([])) == []


def test_multimodal_batches_by_count_without_tokenize() -> None:
    ef = _make_ef("voyage-multimodal-3")
    n = BATCH_SIZE + 3
    texts = [f"t{i}" for i in range(n)]
    batches = list(ef._build_batches(texts))
    assert [len(b) for b in batches] == [BATCH_SIZE, 3]
    ef._client.tokenize.assert_not_called()


# --- Contextualized document path ------------------------------------------


def test_context_document_path_uses_auto_chunking() -> None:
    ef = _make_ef("voyage-context-4")
    _mock_token_counts(ef, [1, 1])
    ef._client.contextualized_embed.return_value = _context_result(
        [[0.1, 0.2], [0.3, 0.4]]
    )

    out = ef._embed_with_batching(["doc a", "doc b"])

    assert out == [[0.1, 0.2], [0.3, 0.4]]
    kwargs = ef._client.contextualized_embed.call_args.kwargs
    # Flat list of documents, NOT one document's chunks (inputs=[batch]).
    assert kwargs["inputs"] == ["doc a", "doc b"]
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32_000
    ef._client.embed.assert_not_called()


def test_context_document_path_is_default_input_type() -> None:
    """No explicit input_type is still treated as a document (auto-chunking on)."""
    ef = _make_ef("voyage-context-4")
    _mock_token_counts(ef, [1])
    ef._client.contextualized_embed.return_value = _context_result([[0.5]])

    ef._embed_with_batching(["doc"])

    kwargs = ef._client.contextualized_embed.call_args.kwargs
    assert kwargs["enable_auto_chunking"] is True
    assert kwargs["chunk_size"] == 32_000


# --- Contextualized query path ---------------------------------------------


def test_context_query_path_disables_auto_chunking() -> None:
    ef = _make_ef("voyage-context-4", input_type="query")
    _mock_token_counts(ef, [1, 1])
    ef._client.contextualized_embed.return_value = _context_result(
        [[0.1], [0.2]]
    )

    out = ef._embed_with_batching(["q1", "q2"])

    assert out == [[0.1], [0.2]]
    kwargs = ef._client.contextualized_embed.call_args.kwargs
    assert kwargs["inputs"] == ["q1", "q2"]
    # API rejects auto-chunking for queries: disabled and no chunk_size sent.
    assert kwargs["enable_auto_chunking"] is False
    assert "chunk_size" not in kwargs
    assert kwargs["input_type"] == "query"


# --- Plain text path -------------------------------------------------------


def test_regular_path_uses_embed() -> None:
    ef = _make_ef("voyage-3.5")
    _mock_token_counts(ef, [1, 1])
    embed_result = MagicMock()
    embed_result.embeddings = [[1.0, 2.0], [3.0, 4.0]]
    ef._client.embed.return_value = embed_result

    out = ef._embed_with_batching(["a", "b"])

    assert out == [[1.0, 2.0], [3.0, 4.0]]
    ef._client.embed.assert_called_once()
    ef._client.contextualized_embed.assert_not_called()


def test_single_string_input_is_wrapped() -> None:
    ef = _make_ef("voyage-3.5")
    _mock_token_counts(ef, [1])
    embed_result = MagicMock()
    embed_result.embeddings = [[1.0, 2.0]]
    ef._client.embed.return_value = embed_result

    out = ef("hello")

    assert list(out[0]) == [1.0, 2.0]


def test_embed_empty_returns_empty() -> None:
    ef = _make_ef("voyage-3.5")
    assert ef._embed_with_batching([]) == []


def test_count_tokens() -> None:
    ef = _make_ef("voyage-3.5")
    _mock_token_counts(ef, [2, 5])
    assert ef.count_tokens(["a", "bb"]) == [2, 5]
    assert ef.count_tokens([]) == []
