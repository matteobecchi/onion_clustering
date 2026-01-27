"""Classes containing data and parameters for Onion clustering."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class OnionParams:
    """Contains the hyperparameters of the Onion clustering."""

    bins: Literal["auto"] | int
    number_of_sigmas: float
    max_area_overlap: float

    def __post_init__(self):
        if isinstance(self.bins, int) and self.bins <= 1:
            raise ValueError("bins must be greater than 1")

        if self.number_of_sigmas <= 0:
            raise ValueError("number_of_sigmas must be positive")

        if not (0.0 <= self.max_area_overlap <= 1.0):
            raise ValueError("max_area_overlap must be between 0 and 1")


@dataclass
class OnionData:
    """Contains the data to cluster, and their labels."""

    data: NDArray[np.float64]
    labels: NDArray[np.int64] | None = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = np.full(
                self.data.shape[:2],
                -1.0,
                dtype=np.int64,
            )
