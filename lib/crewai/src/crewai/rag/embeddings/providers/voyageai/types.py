"""Type definitions for Voyage AI by MongoDB embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class VoyageAIProviderConfig(TypedDict, total=False):
    """Configuration for Voyage AI by MongoDB provider."""

    api_key: str
    model: Annotated[str, "voyage-2"]
    input_type: Literal["document", "query"]
    truncation: Annotated[bool, True]
    output_dtype: str
    output_dimension: int
    max_retries: Annotated[int, 0]
    timeout: float


class VoyageAIProviderSpec(TypedDict):
    """Voyage AI by MongoDB provider specification."""

    provider: Required[Literal["voyageai"]]
    config: VoyageAIProviderConfig
