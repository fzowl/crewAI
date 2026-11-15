"""Tests for the Voyage AI by MongoDB embedding function."""

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.providers.voyageai.embedding_callable import (
    MAX_INPUTS_PER_REQUEST,
    VoyageAIEmbeddingFunction,
    split_into_batches,
)


def _vector(text: str) -> list[float]:
    return [float(len(text))]


def _fake_client() -> MagicMock:
    """A client whose token count is the number of words in a text."""
    client = MagicMock()
    client.tokenize.side_effect = lambda texts, model: [
        SimpleNamespace(ids=text.split()) for text in texts
    ]
    client.embed.side_effect = lambda texts, **kwargs: SimpleNamespace(
        embeddings=[_vector(t) for t in texts]
    )

    def contextualized_embed(inputs: list[Any], **kwargs: Any) -> SimpleNamespace:
        docs = [[d] if isinstance(d, str) else d for d in inputs]
        return SimpleNamespace(
            results=[
                SimpleNamespace(embeddings=[_vector(chunk) for chunk in doc])
                for doc in docs
            ]
        )

    client.contextualized_embed.side_effect = contextualized_embed
    return client


@pytest.fixture
def client():
    fake = _fake_client()
    voyageai = MagicMock()
    voyageai.Client.return_value = fake
    with patch.dict(sys.modules, {"voyageai": voyageai}):
        yield fake


def _function(**config: Any) -> VoyageAIEmbeddingFunction:
    return VoyageAIEmbeddingFunction(api_key="key", **config)


class TestSplitIntoBatches:
    def test_splits_at_token_limit(self):
        assert split_into_batches([5, 5, 5, 5], 10) == [range(0, 2), range(2, 4)]

    def test_input_over_the_limit_gets_its_own_batch(self):
        assert split_into_batches([1, 50, 1], 10) == [
            range(0, 1),
            range(1, 2),
            range(2, 3),
        ]

    def test_splits_at_input_count_limit(self):
        batches = split_into_batches([0] * (MAX_INPUTS_PER_REQUEST + 1), None)

        assert [len(b) for b in batches] == [MAX_INPUTS_PER_REQUEST, 1]

    def test_batches_cover_every_input_once_in_order(self):
        counts = [3, 9, 1, 1, 7, 2, 8, 4]

        flat = [i for b in split_into_batches(counts, 10) for i in b]

        assert flat == list(range(len(counts)))


class TestEmbed:
    def test_batches_requests_and_keeps_input_order(self, client):
        # voyage-4-large allows 120K tokens per request.
        words = " ".join(["w"] * 70_000)
        texts = [words, words + " x", "short"]

        result = _function(model="voyage-4-large")(texts)

        assert [c.kwargs["texts"] for c in client.embed.call_args_list] == [
            [texts[0]],
            [texts[1], texts[2]],
        ]
        assert result == [_vector(t) for t in texts]

    def test_each_model_uses_its_documented_limit(self, client):
        words = " ".join(["w"] * 200_000)

        _function(model="voyage-4-lite")([words, words])
        lite_requests = client.embed.call_count
        client.embed.reset_mock()
        _function(model="voyage-4")([words, words])

        assert lite_requests == 1  # 1M limit
        assert client.embed.call_count == 2  # 320K limit

    def test_model_without_documented_token_limit_is_not_tokenized(self, client):
        result = _function(model="voyage-code-4")(["a", "b"])

        client.tokenize.assert_not_called()
        assert result == [_vector("a"), _vector("b")]

    def test_forwards_output_options(self, client):
        _function(model="voyage-4", input_type="query", output_dimension=256)(["a"])

        kwargs = client.embed.call_args.kwargs
        assert kwargs["input_type"] == "query"
        assert kwargs["output_dimension"] == 256


class TestContextualized:
    def test_document_mode_auto_chunks_a_flat_list(self, client):
        result = _function(model="voyage-context-4", input_type="document")(
            ["doc one", "doc two"]
        )

        client.embed.assert_not_called()
        kwargs = client.contextualized_embed.call_args.kwargs
        assert kwargs["inputs"] == ["doc one", "doc two"]
        assert kwargs["enable_auto_chunking"] is True
        assert kwargs["chunk_size"] == 32_000
        assert result == [_vector("doc one"), _vector("doc two")]

    @pytest.mark.parametrize("input_type", ["query", None])
    def test_without_document_type_nothing_is_auto_chunked(self, client, input_type):
        config: dict[str, Any] = {"model": "voyage-context-3"}
        if input_type:
            config["input_type"] = input_type

        result = _function(**config)(["q1", "q2"])

        kwargs = client.contextualized_embed.call_args.kwargs
        assert kwargs["inputs"] == [["q1"], ["q2"]]
        assert "enable_auto_chunking" not in kwargs
        assert "chunk_size" not in kwargs
        assert result == [_vector("q1"), _vector("q2")]

    def test_document_batches_split_at_120k_tokens(self, client):
        words = " ".join(["w"] * 70_000)

        _function(model="voyage-context-4", input_type="document")([words, words])

        assert client.contextualized_embed.call_count == 2

    def test_query_batches_split_at_32k_tokens(self, client):
        words = " ".join(["w"] * 20_000)

        _function(model="voyage-context-4", input_type="query")([words, words])

        assert client.contextualized_embed.call_count == 2

    def test_input_split_into_several_chunks_is_an_error(self, client):
        client.contextualized_embed.side_effect = None
        client.contextualized_embed.return_value = SimpleNamespace(
            results=[SimpleNamespace(embeddings=[[0.1], [0.2]])]
        )

        with pytest.raises(ValueError, match="split an input into 2 chunks"):
            _function(model="voyage-context-4", input_type="document")(["long doc"])


def test_unloadable_tokenizer_is_reported_with_the_model_name(client):
    client.tokenize.side_effect = OSError("repository not found")

    with pytest.raises(ValueError, match="voyage-nope-9"):
        _function(model="voyage-nope-9")(["a"])

    client.embed.assert_not_called()
