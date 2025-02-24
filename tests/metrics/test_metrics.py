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

from quick.circuit import QiskitCircuit
from quick.metrics import (
    calculate_entanglement_range,
    calculate_shannon_entropy,
    calculate_entanglement_entropy
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

    @pytest.mark.parametrize("data", [
        np.array([1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.5, 0.5j, -0.5j, 0.5]),
        np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)* 1j]),
        np.array([1/np.sqrt(2)] + (14 * [0]) + [1/np.sqrt(2) * 1j]),
    ])
    def test_calculate_entanglement_entropy(
            self,
            data: NDArray[np.complex128]
        ) -> None:
        """ Test the `calculate_entanglement_entropy` method.

        Parameters
        ----------
        data : NDArray[np.complex128]
            The statevector of the circuit.
        """
        assert_almost_equal(0.0, calculate_entanglement_entropy(data))