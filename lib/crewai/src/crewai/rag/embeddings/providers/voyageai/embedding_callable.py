"""VoyageAI by MongoDB embedding function implementation."""

from typing import cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


# Per-chunk token limit accepted by the contextualized_embed endpoint.
_CONTEXTUALIZED_CHUNK_SIZE = 32000


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for VoyageAI by MongoDB models."""

    def __init__(self, **kwargs: Unpack[VoyageAIProviderConfig]) -> None:
        """Initialize VoyageAI by MongoDB embedding function.

        Args:
            **kwargs: Configuration parameters for VoyageAI by MongoDB.
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

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.
        """

        if isinstance(input, str):
            input = [input]

        model = self._config.get("model", "voyage-2")
        if model.startswith("voyage-context"):
            return self._contextualized_embed(input, model)

        result = self._client.embed(
            texts=input,
            model=model,
            input_type=self._config.get("input_type"),
            truncation=self._config.get("truncation", True),
            output_dtype=self._config.get("output_dtype"),
            output_dimension=self._config.get("output_dimension"),
        )

        return cast(Embeddings, result.embeddings)

    def _contextualized_embed(self, input: Documents, model: str) -> Embeddings:
        """Embed documents with a contextualized (``voyage-context-*``) model.

        Each input string is embedded as its own independent document: the batch
        is sent to ``contextualized_embed`` as a flat ``list[str]`` with
        ``enable_auto_chunking=True`` and ``chunk_size=32000``, so every input
        resolves to a single chunk and therefore a single vector. This keeps the
        result one-embedding-per-input, matching the ChromaDB contract.

        Auto-chunking requires ``input_type="document"`` and is rejected by the
        API for queries, so the query path disables it and drops ``chunk_size``.

        Args:
            input: List of documents to embed.
            model: The ``voyage-context-*`` model name.

        Returns:
            List of embedding vectors, one per input.
        """
        input_type = self._config.get("input_type")
        is_query = input_type == "query"

        kwargs: dict = {
            "inputs": list(input),
            "model": model,
            "output_dtype": self._config.get("output_dtype"),
            "output_dimension": self._config.get("output_dimension"),
        }
        if is_query:
            kwargs["input_type"] = "query"
            kwargs["enable_auto_chunking"] = False
        else:
            # Auto-chunking is only valid with input_type="document".
            kwargs["input_type"] = "document"
            kwargs["enable_auto_chunking"] = True
            kwargs["chunk_size"] = _CONTEXTUALIZED_CHUNK_SIZE

        result = self._client.contextualized_embed(**kwargs)

        # Each input maps to one result; take its first (and only) chunk vector.
        embeddings = [item.embeddings[0] for item in result.results]
        return cast(Embeddings, embeddings)
