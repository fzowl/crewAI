"""VoyageAI by MongoDB embedding function implementation."""

from typing import Any, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


DEFAULT_MODEL = "voyage-3.5"


def _is_contextualized_model(model: str) -> bool:
    """Return True if the model uses the contextualized embeddings endpoint.

    Contextualized models (e.g. ``voyage-context-4``) embed each chunk with
    awareness of the surrounding chunks and must be called through the
    ``contextualized_embed`` endpoint instead of ``embed``.
    """
    return model.startswith("voyage-context-")


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for VoyageAI by MongoDB models."""

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

        Contextualized models (``voyage-context-*``) are routed through the
        ``contextualized_embed`` endpoint; every other model uses ``embed``.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors, one per input document.
        """

        if isinstance(input, str):
            input = [input]

        model = self._config.get("model", DEFAULT_MODEL)

        if _is_contextualized_model(model):
            # ChromaDB passes independent documents, so each one becomes its own
            # single-chunk group to keep contextualization scoped per document.
            result = self.contextualized_embed(
                [[text] for text in input], model=model
            )
            return cast(
                Embeddings, [chunk.embeddings[0] for chunk in result.results]
            )

        result = self._client.embed(
            texts=input,
            model=model,
            input_type=self._config.get("input_type"),
            truncation=self._config.get("truncation", True),
            output_dtype=self._config.get("output_dtype"),
            output_dimension=self._config.get("output_dimension"),
        )

        return cast(Embeddings, result.embeddings)

    def contextualized_embed(
        self,
        inputs: list[list[str]] | list[str],
        model: str | None = None,
    ) -> Any:
        """Call the VoyageAI by MongoDB contextualized embeddings endpoint.

        Follows the official ``contextualized_embed`` spec, accepting both
        supported ``inputs`` formats:
        https://docs.voyageai.com/docs/contextualized-chunk-embeddings

        Args:
            inputs: Either a list of documents where each document is a list of
                its chunks (``list[list[str]]``), or a single document's chunks
                as a flat ``list[str]``. A flat list is treated as one document.
            model: Contextualized model name. Defaults to the configured model.

        Returns:
            The raw VoyageAI contextualized embeddings response, whose
            ``results`` hold per-chunk embeddings grouped by input document.
        """
        normalized: list[list[str]]
        if inputs and isinstance(inputs[0], str):
            normalized = [cast("list[str]", inputs)]
        else:
            normalized = cast("list[list[str]]", inputs)

        return self._client.contextualized_embed(
            inputs=normalized,
            model=model or self._config.get("model", DEFAULT_MODEL),
            input_type=self._config.get("input_type"),
            output_dtype=self._config.get("output_dtype", "float"),
            output_dimension=self._config.get("output_dimension"),
        )
