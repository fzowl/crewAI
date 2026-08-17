"""VoyageAI embedding function implementation."""

from collections.abc import Callable, Generator
from typing import cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


# Per-request total token limits for VoyageAI models, used to build token-aware
# batches. Source: https://docs.voyageai.com/docs/embeddings (model overview) and
# https://docs.voyageai.com/docs/contextualized-chunk-embeddings.
# Unlisted models fall back to DEFAULT_TOTAL_TOKEN_LIMIT.
VOYAGE_TOTAL_TOKEN_LIMITS = {
    # Voyage 4 series (current)
    "voyage-4-large": 120_000,
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
    "voyage-code-4": 120_000,
    "voyage-4-nano": 120_000,  # per-request cap not published; conservative default
    # Domain-specialized models
    "voyage-finance-2": 120_000,
    "voyage-law-2": 120_000,
    # Voyage 3 / 3.5 series
    "voyage-3-large": 120_000,
    "voyage-3.5": 320_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-3": 120_000,
    "voyage-multilingual-2": 120_000,
    # Voyage 2 series (legacy)
    "voyage-large-2-instruct": 120_000,
    "voyage-large-2": 120_000,
    "voyage-2": 320_000,
    "voyage-code-2": 120_000,
    # Contextualized chunk embeddings (120K total across inputs with auto-chunking)
    "voyage-context-4": 120_000,
    "voyage-context-3": 120_000,
    # Multimodal
    "voyage-multimodal-3": 32_000,
}

# Fallback token limit for models not present in VOYAGE_TOTAL_TOKEN_LIMITS.
DEFAULT_TOTAL_TOKEN_LIMIT = 120_000

# Maximum number of inputs VoyageAI accepts in a single request.
BATCH_SIZE = 1000

# Target chunk size (in tokens) for contextualized document embedding.
# Each document input is <= 32K tokens, so with auto-chunking it resolves to
# exactly one chunk and therefore exactly one embedding per input.
CONTEXT_CHUNK_SIZE = 32_000


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for VoyageAI models.

    Supports plain text models (``embed``), contextualized chunk models
    (``contextualized_embed``, e.g. ``voyage-context-4``) and multimodal models
    (``multimodal_embed``). Requests are split into token-aware batches so a
    single call to ``__call__`` can embed arbitrarily many texts without
    exceeding the per-model token limit.

    Contextualized models embed each input string as its OWN independent
    document: the batch is sent as a flat ``list[str]`` with auto-chunking so
    every input resolves to a single chunk and a single vector. Cross-input
    contextualization is intentionally not used, because generic callers pass
    unrelated texts.
    """

    def __init__(self, **kwargs: Unpack[VoyageAIProviderConfig]) -> None:
        """Initialize VoyageAI embedding function.

        Args:
            **kwargs: Configuration parameters for VoyageAI.
        """
        try:
            import voyageai

        except ImportError as e:
            raise ImportError(
                "voyageai is required for voyageai embeddings. "
                "Install it with: uv add voyageai"
            ) from e
        self._config = kwargs
        self._model = kwargs.get("model", "voyage-2")
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

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors, one per input document.
        """
        if isinstance(input, str):
            input = [input]

        embeddings = self._embed_with_batching(list(input))

        return cast(Embeddings, embeddings)

    def _is_context_model(self) -> bool:
        """Check if the model is a contextualized embedding model."""
        return "context" in self._model

    def _is_multimodal_model(self) -> bool:
        """Check if the model is a multimodal embedding model."""
        return "multimodal" in self._model

    def get_token_limit(self) -> int:
        """Return the per-request total token limit for the current model."""
        return VOYAGE_TOTAL_TOKEN_LIMITS.get(self._model, DEFAULT_TOTAL_TOKEN_LIMIT)

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Count tokens for each text using the VoyageAI tokenize API.

        Args:
            texts: List of texts to count tokens for.

        Returns:
            List of token counts, one per input text.
        """
        if not texts:
            return []

        token_lists = self._client.tokenize(texts, model=self._model)
        return [len(token_list) for token_list in token_lists]

    def _build_batches_by_count(
        self, texts: list[str]
    ) -> Generator[list[str], None, None]:
        """Yield batches based on item count only (for multimodal models).

        The tokenize API does not support multimodal models, so these fall back
        to a fixed item-count cap.

        Args:
            texts: List of texts to batch.

        Yields:
            Batches of texts.
        """
        for i in range(0, len(texts), BATCH_SIZE):
            yield texts[i : i + BATCH_SIZE]

    def _build_batches(self, texts: list[str]) -> Generator[list[str], None, None]:
        """Yield token-aware batches of texts.

        Batches are bounded by both the per-model total token limit and the
        fixed ``BATCH_SIZE`` item cap. A single text larger than the token limit
        is emitted on its own rather than dropped.

        Args:
            texts: List of texts to batch.

        Yields:
            Batches of texts.
        """
        if not texts:
            return

        # Multimodal models can't be tokenized; batch by item count only.
        if self._is_multimodal_model():
            yield from self._build_batches_by_count(texts)
            return

        max_tokens_per_batch = self.get_token_limit()
        token_counts = self.count_tokens(texts)

        current_batch: list[str] = []
        current_batch_tokens = 0

        for text, n_tokens in zip(texts, token_counts, strict=True):
            # Start a new batch if adding this text would exceed either the item
            # cap or the token cap. A non-empty guard keeps an oversized single
            # text in its own batch instead of dropping it.
            if current_batch and (
                len(current_batch) >= BATCH_SIZE
                or current_batch_tokens + n_tokens > max_tokens_per_batch
            ):
                yield current_batch
                current_batch = []
                current_batch_tokens = 0

            current_batch.append(text)
            current_batch_tokens += n_tokens

        if current_batch:
            yield current_batch

    def _get_embed_function(self) -> Callable[[list[str]], list[list[float]]]:
        """Return the embedding callable matching the configured model type."""
        model_name = self._model

        if self._is_context_model():

            def embed_batch_context(batch: list[str]) -> list[list[float]]:
                # Embed each input as its own independent document. The batch is
                # a flat list[str]; auto-chunking with a 32K chunk size makes
                # every input resolve to exactly one chunk -> one embedding.
                # Auto-chunking is rejected by the API for input_type="query",
                # so it is only enabled for document-side inputs.
                input_type = self._config.get("input_type")
                enable_auto_chunking = input_type != "query"

                kwargs: dict[str, object] = {
                    "inputs": batch,
                    "model": model_name,
                    "input_type": input_type,
                    "output_dimension": self._config.get("output_dimension"),
                    "output_dtype": self._config.get("output_dtype"),
                    "enable_auto_chunking": enable_auto_chunking,
                }
                if enable_auto_chunking:
                    kwargs["chunk_size"] = CONTEXT_CHUNK_SIZE

                result = self._client.contextualized_embed(**kwargs)
                # One result per input document, each with a single embedding.
                return [list(res.embeddings[0]) for res in result.results]

            return embed_batch_context

        if self._is_multimodal_model():

            def embed_batch_multimodal(batch: list[str]) -> list[list[float]]:
                # Multimodal API expects a list of content lists; for text-only
                # inputs that is [[text1], [text2], ...].
                inputs = [[text] for text in batch]
                result = self._client.multimodal_embed(
                    inputs=inputs,
                    model=model_name,
                    input_type=self._config.get("input_type"),
                    truncation=self._config.get("truncation", True),
                )
                return [list(emb) for emb in result.embeddings]

            return embed_batch_multimodal

        def embed_batch_regular(batch: list[str]) -> list[list[float]]:
            result = self._client.embed(
                texts=batch,
                model=model_name,
                input_type=self._config.get("input_type"),
                truncation=self._config.get("truncation", True),
                output_dimension=self._config.get("output_dimension"),
                output_dtype=self._config.get("output_dtype"),
            )
            return [list(emb) for emb in result.embeddings]

        return embed_batch_regular

    def _embed_with_batching(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with token-aware batching.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one per input text and in input order.
        """
        if not texts:
            return []

        embed_fn = self._get_embed_function()

        all_embeddings: list[list[float]] = []
        for batch in self._build_batches(texts):
            all_embeddings.extend(embed_fn(batch))

        return all_embeddings
