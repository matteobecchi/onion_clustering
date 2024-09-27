"""onion-clustering package."""

from onion_clustering._internal.onion_multi import (
    OnionMulti,
    onion_multi,
)
from onion_clustering._internal.onion_uni import (
    OnionUni,
    onion_uni,
)

__all__ = [
    "onion_uni",
    "OnionUni",
    "onion_multi",
    "OnionMulti",
]
