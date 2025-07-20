# Copyright 2023-2025 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/quick/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ["TestMetrics"]

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal
import pytest
from scipy.stats import unitary_group

from quick.circuit import QiskitCircuit
from quick.metrics import (
    calculate_entanglement_range,
    calculate_shannon_entropy,
    calculate_entanglement_entropy,
    calculate_hilbert_schmidt_test
)


class TestMetrics:
    """ `tests.circuit.metrics.TestMetrics` class to test the `quick.circuit.metrics`
    module.
    """
    def test_calculate_entanglement_range(self) -> None:
        """ Test the `calculate_entanglement_range` method.
        """
        qc = QiskitCircuit(6)
        qc.H(0)
        qc.CX(0, 1)

        qc.H(3)
        qc.CX(3, 4)
        qc.CX(4, 5)

        entanglements = calculate_entanglement_range(qc.get_statevector())
        assert entanglements == [(0, 1), (2, 2), (3, 5)]

    def test_calculate_shannon_entropy(self) -> None:
        """ Test the `calculate_shannon_entropy` method.
        """
        data = np.array([0.5, 0.3, 0.07, 0.1, 0.03])

        assert_almost_equal(1.7736043871504037, calculate_shannon_entropy(data))

    @pytest.mark.parametrize("data, expected", [
        (np.array([1, 0]), 0.0),
        (np.array([0, 1, 0, 0]), 0.0),
        (np.array([0.5, 0.5, 0.5, 0.5]), 0.0),
        (np.array([0.5, 0.5j, -0.5j, 0.5]), 0.0),
        (np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)* 1j]), 0.0),
        (np.array([1/np.sqrt(2)] + (14 * [0]) + [1/np.sqrt(2) * 1j]), 0.0),
        (
            np.array([
                [0.84048555+0.j, 0.0510054-0.02325157j],
                [0.0510054+0.02325157j, 0.15951445+0.j]
            ], dtype=np.complex128),
            0.6220438669480641
        ),
        (
            np.array([
                [0.16521093+0.j, -0.08915021+0.07244625j, -0.14670846-0.10748953j, -0.03544851+0.106916j],
                [-0.08915021-0.07244625j, 0.28432666+0.j, -0.01778044+0.15666538j, -0.03049137-0.01306784j],
                [-0.14670846+0.10748953j, -0.01778044-0.15666538j, 0.2941435 +0.j, -0.06158772-0.08660825j],
                [-0.03544851-0.106916j, -0.03049137+0.01306784j, -0.06158772+0.08660825j, 0.2563189+0.j]
            ], dtype=np.complex128),
            1.3705180586061732
        )
    ])
    def test_calculate_entanglement_entropy(
            self,
            data: NDArray[np.complex128],
            expected: float
        ) -> None:
        """ Test the `calculate_entanglement_entropy` method.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The statevector of the circuit.
        `expected` : float
            The expected value.
        """
        assert_almost_equal(expected, calculate_entanglement_entropy(data))

    @pytest.mark.parametrize("data", [
        np.array([1, 2, 3], dtype=np.complex128),
        np.array([1, 2, 3, 4], dtype=np.complex128),
        np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.complex128),
        np.array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=np.complex128)
    ])
    def test_calculate_entanglement_entropy_invalid_data(
            self,
            data: NDArray[np.complex128]
        ) -> None:
        """ Test failure of `calculate_entanglement_entropy` with invalid
        data values, which are neither density matrix nor statevector.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data to be tested.
        """
        with pytest.raises(ValueError):
            calculate_entanglement_entropy(data)

    def test_calculate_entanglement_entropy_slope_area_law_case(self) -> None:
        """ Test the `calculate_entanglement_entropy_slope` method with an area-law
        entangled state.
        """
        area_law_slope = np.load("tests/metrics/area_law_slope.npy")
        assert_almost_equal(0.20183287022097673, area_law_slope)

    def test_calculate_entanglement_entropy_slope_volume_law_case(self) -> None:
        """ Test the `calculate_entanglement_entropy_slope` method with a volume-law
        entangled state.
        """
        volume_law_slope = np.load("tests/metrics/volume_law_slope.npy")
        assert_almost_equal(1.0, volume_law_slope)

    def test_calculate_hilbert_schmidt_test(self) -> None:
        """ Test the `calculate_hilbert_schmidt_test` method.
        """
        unitary = unitary_group.rvs(4).astype(np.complex128)
        assert_almost_equal(1.0, calculate_hilbert_schmidt_test(unitary, unitary))

    def test_calculate_hilbert_schmidt_fail(self) -> None:
        """ Test the `calculate_hilbert_schmidt_test` method with invalid inputs.
        """
        unitary = unitary_group.rvs(4).astype(np.complex128)

        with pytest.raises(ValueError):
            calculate_hilbert_schmidt_test(unitary, np.zeros((4, 4))) # type: ignore
        with pytest.raises(ValueError):
            calculate_hilbert_schmidt_test(unitary, np.zeros((4, 3))) # type: ignore
        with pytest.raises(ValueError):
            calculate_hilbert_schmidt_test(np.zeros((4, 4)), unitary) # type: ignore