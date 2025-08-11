"""Pytest for onion_uni_smooth and OnionUniSmooth."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tropea_clustering import OnionUniSmooth, onion_uni_smooth

# ---------------- Fixtures ----------------


@pytest.fixture(scope="module")
def input_data() -> np.ndarray:
    rng = np.random.default_rng(12345)
    input_data = np.array(
        [
            np.concatenate(
                (rng.normal(0.0, 0.1, 500), rng.normal(1.0, 0.1, 500))
            )
            for _ in range(100)
        ]
    )

    return input_data


# ---------------- Tests ----------------


def test_onion_uni_smooth(input_data: np.ndarray):
    delta_t = 10

    # Test class interface
    on_cl = OnionUniSmooth(delta_t)
    on_cl.set_params(bins="auto", number_of_sigmas=3.0)
    _ = on_cl.get_params()
    _ = on_cl.fit_predict(input_data)

    # Test functional interface
    state_list, labels = onion_uni_smooth(input_data, delta_t)
    _ = state_list[0].get_attributes()

    # Check clustering output
    this_dir = Path(__file__).parent
    expected = np.load(this_dir / "output_uni_smooth/labels.npy")
    assert_array_equal(labels, expected)


def test_wrong_input():
    on_cl = OnionUniSmooth(delta_t=10)

    input_data = np.ones(10)  # 1D array
    with pytest.raises(ValueError, match="Expected 2-dimensional input data."):
        on_cl.fit(input_data)

    input_data = np.empty((0, 5))  # empty array
    with pytest.raises(ValueError, match="Empty dataset."):
        on_cl.fit(input_data)

    input_data = np.zeros((100, 1))  # just one frame
    with pytest.raises(ValueError, match="n_frames = 1."):
        on_cl.fit(input_data)

    input_data = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
    with pytest.raises(ValueError, match="Complex data not supported."):
        on_cl.fit(input_data)
