"""Voyage AI by MongoDB embedding function implementation."""

from collections.abc import Sequence
from typing import Any, cast

from typing_extensions import Unpack

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.types import Documents, Embeddings
from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


MAX_INPUTS_PER_REQUEST = 1000

# Maximum total tokens per ``/v1/embeddings`` request. Models that are not
# listed use the documented limit for "all other models". ``None`` means the
# Voyage docs publish no per-request token limit for the model, so only the
# input count limit is applied.
DEFAULT_REQUEST_TOKEN_LIMIT = 120_000
REQUEST_TOKEN_LIMITS: dict[str, int | None] = {
    "voyage-4-lite": 1_000_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-4": 320_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-4-large": 120_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-code-4": None,
}

CONTEXTUALIZED_MODELS = frozenset({"voyage-context-4", "voyage-context-3"})
# ``/v1/contextualizedembeddings`` request limits: 120K tokens with
# auto-chunking, 32K without it.
CONTEXTUALIZED_AUTO_CHUNK_TOKEN_LIMIT = 120_000
CONTEXTUALIZED_TOKEN_LIMIT = 32_000
CONTEXTUALIZED_CHUNK_SIZE = 32_000


def split_into_batches(
    token_counts: Sequence[int], max_tokens: int | None
) -> list[range]:
    """Split consecutive inputs into batches that fit the request limits.

    A batch holds at most ``MAX_INPUTS_PER_REQUEST`` inputs and, when
    ``max_tokens`` is set, at most ``max_tokens`` tokens. An input that exceeds
    ``max_tokens`` on its own gets a batch of its own, so the API reports the
    error instead of the input being dropped.

    Args:
        token_counts: Token count of each input, in input order.
        max_tokens: Maximum total tokens per batch, or None for no token limit.

    Returns:
        Index ranges covering all inputs, in order.
    """
    batches: list[range] = []
    start = 0
    batch_tokens = 0
    for index, count in enumerate(token_counts):
        batch_size = index - start
        if batch_size and (
            batch_size >= MAX_INPUTS_PER_REQUEST
            or (max_tokens is not None and batch_tokens + count > max_tokens)
        ):
            batches.append(range(start, index))
            start = index
            batch_tokens = 0
        batch_tokens += count
    if start < len(token_counts):
        batches.append(range(start, len(token_counts)))
    return batches


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for Voyage AI by MongoDB models."""

    def __init__(self, **kwargs: Unpack[VoyageAIProviderConfig]) -> None:
        """Initialize Voyage AI by MongoDB embedding function.

        Args:
            **kwargs: Configuration parameters for Voyage AI by MongoDB.
        """
        try:
            import voyageai

        except ImportError as e:
            raise ImportError(
                "voyageai is required for voyageai embeddings. "
                "Install it with: uv add voyageai"
            ) from e
        self._config = kwargs
        self._client = voyageai.Client(  # type: ignore[attr-defined]
            api_key=kwargs["api_key"],
            max_retries=kwargs.get("max_retries", 0),
            timeout=kwargs.get("timeout"),
        )

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "voyageai"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Inputs are split into requests that respect the model's per-request
        token limit and the 1000 input limit; the vectors come back in the
        order of ``input``.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors, one per input.

        Raises:
            ValueError: If an input is not a string, or a contextualized model
                splits one input into several chunks, which cannot be returned
                as a single vector.
        """

        if isinstance(input, str):
            input = [input]
        if not all(isinstance(text, str) for text in input):
            raise ValueError(
                "Voyage AI embeddings expect a flat list of strings, one per "
                "document; pre-chunked list[list[str]] input is not supported."
            )

        model = self._config.get("model", "voyage-2")
        if model in CONTEXTUALIZED_MODELS:
            return self._embed_contextualized(input, model)
        return self._embed(input, model)

    def _batches(
        self, input: Documents, model: str, max_tokens: int | None
    ) -> list[Documents]:
        """Split ``input`` into batches using the SDK's token counts.

        The tokenizer is downloaded from Hugging Face, so it is only used when
        the inputs could exceed the token limit: a token covers at least one
        byte, plus up to two special tokens per input.
        """
        if max_tokens is None or (
            sum(len(text.encode()) + 2 for text in input) <= max_tokens
        ):
            token_counts = [0] * len(input)
        else:
            try:
                encodings = self._client.tokenize(input, model)
            except Exception as e:
                raise ValueError(
                    f"Could not load the Voyage tokenizer for model {model!r}; "
                    "check the model name and access to huggingface.co."
                ) from e
            token_counts = [len(encoding.ids) for encoding in encodings]
        return [
            [input[i] for i in batch]
            for batch in split_into_batches(token_counts, max_tokens)
        ]

    def _embed(self, input: Documents, model: str) -> Embeddings:
        max_tokens = REQUEST_TOKEN_LIMITS.get(model, DEFAULT_REQUEST_TOKEN_LIMIT)
        embeddings: list[Any] = []
        for batch in self._batches(input, model, max_tokens):
            result = self._client.embed(
                texts=batch,
                model=model,
                input_type=self._config.get("input_type"),
                truncation=self._config.get("truncation", True),
                output_dtype=self._config.get("output_dtype"),
                output_dimension=self._config.get("output_dimension"),
            )
            embeddings.extend(result.embeddings)
        return cast(Embeddings, embeddings)

    def _embed_contextualized(self, input: Documents, model: str) -> Embeddings:
        """Embed each input as its own document with a contextualized model.

        With ``input_type="document"`` the API chunks each document itself
        (``list[str]``, one string per document). The API accepts auto-chunking
        only for documents, so any other input type sends each input as a
        single-chunk document.
        """
        auto_chunk = self._config.get("input_type") == "document"
        kwargs: dict[str, Any] = {}
        if auto_chunk:
            kwargs["enable_auto_chunking"] = True
            kwargs["chunk_size"] = CONTEXTUALIZED_CHUNK_SIZE
            max_tokens = CONTEXTUALIZED_AUTO_CHUNK_TOKEN_LIMIT
        else:
            max_tokens = CONTEXTUALIZED_TOKEN_LIMIT

        embeddings: list[Any] = []
        for batch in self._batches(input, model, max_tokens):
            inputs: list[Any] = (
                list(batch) if auto_chunk else [[text] for text in batch]
            )
            result = self._client.contextualized_embed(
                inputs=inputs,
                model=model,
                input_type=self._config.get("input_type"),
                output_dtype=self._config.get("output_dtype"),
                output_dimension=self._config.get("output_dimension"),
                **kwargs,
            )
            for text, item in zip(batch, result.results, strict=True):
                if len(item.embeddings) != 1:
                    raise ValueError(
                        f"{model} split an input into {len(item.embeddings)} "
                        f"chunks (starting with {text[:40]!r}); inputs must "
                        "fit in one 32,000 token chunk to get a single vector."
                    )
                embeddings.append(item.embeddings[0])
        return cast(Embeddings, embeddings)
