# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/QICKIT/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = [
    "test_qft_no_swap_no_inverse",
    "test_qft_swap_no_inverse",
    "test_qft_no_swap_inverse",
    "test_qft_swap_inverse"
]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
import pytest
from typing import Type

from qickit.circuit import Circuit, CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit

from tests.circuit.gate_utils import (
    qft_no_swap_no_inverse_approx0_5qubits,
    qft_no_swap_no_inverse_approx0_6qubits,
    qft_no_swap_no_inverse_approx0_7qubits,
    qft_no_swap_no_inverse_approx0_8qubits,
    qft_no_swap_no_inverse_approx1_5qubits,
    qft_no_swap_no_inverse_approx1_6qubits,
    qft_no_swap_no_inverse_approx1_7qubits,
    qft_no_swap_no_inverse_approx1_8qubits,
    qft_no_swap_no_inverse_approx2_5qubits,
    qft_no_swap_no_inverse_approx2_6qubits,
    qft_no_swap_no_inverse_approx2_7qubits,
    qft_no_swap_no_inverse_approx2_8qubits,
    qft_no_swap_no_inverse_approx3_5qubits,
    qft_no_swap_no_inverse_approx3_6qubits,
    qft_no_swap_no_inverse_approx3_7qubits,
    qft_no_swap_no_inverse_approx3_8qubits,
    qft_swap_no_inverse_approx0_5qubits,
    qft_swap_no_inverse_approx0_6qubits,
    qft_swap_no_inverse_approx0_7qubits,
    qft_swap_no_inverse_approx0_8qubits,
    qft_swap_no_inverse_approx1_5qubits,
    qft_swap_no_inverse_approx1_6qubits,
    qft_swap_no_inverse_approx1_7qubits,
    qft_swap_no_inverse_approx1_8qubits,
    qft_swap_no_inverse_approx2_5qubits,
    qft_swap_no_inverse_approx2_6qubits,
    qft_swap_no_inverse_approx2_7qubits,
    qft_swap_no_inverse_approx2_8qubits,
    qft_swap_no_inverse_approx3_5qubits,
    qft_swap_no_inverse_approx3_6qubits,
    qft_swap_no_inverse_approx3_7qubits,
    qft_swap_no_inverse_approx3_8qubits,
    qft_no_swap_inverse_approx0_5qubits,
    qft_no_swap_inverse_approx0_6qubits,
    qft_no_swap_inverse_approx0_7qubits,
    qft_no_swap_inverse_approx0_8qubits,
    qft_no_swap_inverse_approx1_5qubits,
    qft_no_swap_inverse_approx1_6qubits,
    qft_no_swap_inverse_approx1_7qubits,
    qft_no_swap_inverse_approx1_8qubits,
    qft_no_swap_inverse_approx2_5qubits,
    qft_no_swap_inverse_approx2_6qubits,
    qft_no_swap_inverse_approx2_7qubits,
    qft_no_swap_inverse_approx2_8qubits,
    qft_no_swap_inverse_approx3_5qubits,
    qft_no_swap_inverse_approx3_6qubits,
    qft_no_swap_inverse_approx3_7qubits,
    qft_no_swap_inverse_approx3_8qubits,
    qft_swap_inverse_approx0_5qubits,
    qft_swap_inverse_approx0_6qubits,
    qft_swap_inverse_approx0_7qubits,
    qft_swap_inverse_approx0_8qubits,
    qft_swap_inverse_approx1_5qubits,
    qft_swap_inverse_approx1_6qubits,
    qft_swap_inverse_approx1_7qubits,
    qft_swap_inverse_approx1_8qubits,
    qft_swap_inverse_approx2_5qubits,
    qft_swap_inverse_approx2_6qubits,
    qft_swap_inverse_approx2_7qubits,
    qft_swap_inverse_approx2_8qubits,
    qft_swap_inverse_approx3_5qubits,
    qft_swap_inverse_approx3_6qubits,
    qft_swap_inverse_approx3_7qubits,
    qft_swap_inverse_approx3_8qubits
)

# The quantum circuit frameworks
CIRCUIT_FRAMEWORKS = [CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit]


@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
@pytest.mark.parametrize("num_qubits, approximation_degree, expected_unitary", [
    (5, 0, qft_no_swap_no_inverse_approx0_5qubits),
    (6, 0, qft_no_swap_no_inverse_approx0_6qubits),
    (7, 0, qft_no_swap_no_inverse_approx0_7qubits),
    (8, 0, qft_no_swap_no_inverse_approx0_8qubits),
    (5, 1, qft_no_swap_no_inverse_approx1_5qubits),
    (6, 1, qft_no_swap_no_inverse_approx1_6qubits),
    (7, 1, qft_no_swap_no_inverse_approx1_7qubits),
    (8, 1, qft_no_swap_no_inverse_approx1_8qubits),
    (5, 2, qft_no_swap_no_inverse_approx2_5qubits),
    (6, 2, qft_no_swap_no_inverse_approx2_6qubits),
    (7, 2, qft_no_swap_no_inverse_approx2_7qubits),
    (8, 2, qft_no_swap_no_inverse_approx2_8qubits),
    (5, 3, qft_no_swap_no_inverse_approx3_5qubits),
    (6, 3, qft_no_swap_no_inverse_approx3_6qubits),
    (7, 3, qft_no_swap_no_inverse_approx3_7qubits),
    (8, 3, qft_no_swap_no_inverse_approx3_8qubits),
])
def test_qft_no_swap_no_inverse(
        circuit_framework: Type[Circuit],
        num_qubits: int,
        approximation_degree: int,
        expected_unitary: NDArray[np.complex128]
    ) -> None:
    """ Test the quantum Fourier transform without swaps and without inverse.
    """
    qft_circuit = circuit_framework(num_qubits)
    qft_circuit.QFT(
        range(num_qubits),
        approximation_degree=approximation_degree,
        do_swaps=False,
        inverse=False
    )
    assert_almost_equal(qft_circuit.get_unitary(), expected_unitary, 8)

@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
@pytest.mark.parametrize("num_qubits, approximation_degree, expected_unitary", [
    (5, 0, qft_swap_no_inverse_approx0_5qubits),
    (6, 0, qft_swap_no_inverse_approx0_6qubits),
    (7, 0, qft_swap_no_inverse_approx0_7qubits),
    (8, 0, qft_swap_no_inverse_approx0_8qubits),
    (5, 1, qft_swap_no_inverse_approx1_5qubits),
    (6, 1, qft_swap_no_inverse_approx1_6qubits),
    (7, 1, qft_swap_no_inverse_approx1_7qubits),
    (8, 1, qft_swap_no_inverse_approx1_8qubits),
    (5, 2, qft_swap_no_inverse_approx2_5qubits),
    (6, 2, qft_swap_no_inverse_approx2_6qubits),
    (7, 2, qft_swap_no_inverse_approx2_7qubits),
    (8, 2, qft_swap_no_inverse_approx2_8qubits),
    (5, 3, qft_swap_no_inverse_approx3_5qubits),
    (6, 3, qft_swap_no_inverse_approx3_6qubits),
    (7, 3, qft_swap_no_inverse_approx3_7qubits),
    (8, 3, qft_swap_no_inverse_approx3_8qubits),
])
def test_qft_swap_no_inverse(
        circuit_framework: Type[Circuit],
        num_qubits: int,
        approximation_degree: int,
        expected_unitary: NDArray[np.complex128]
    ) -> None:
    """ Test the quantum Fourier transform with swaps and without inverse.
    """
    qft_circuit = circuit_framework(num_qubits)
    qft_circuit.QFT(
        range(num_qubits),
        approximation_degree=approximation_degree,
        do_swaps=True,
        inverse=False
    )
    assert_almost_equal(qft_circuit.get_unitary(), expected_unitary, 8)

@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
@pytest.mark.parametrize("num_qubits, approximation_degree, expected_unitary", [
    (5, 0, qft_no_swap_inverse_approx0_5qubits),
    (6, 0, qft_no_swap_inverse_approx0_6qubits),
    (7, 0, qft_no_swap_inverse_approx0_7qubits),
    (8, 0, qft_no_swap_inverse_approx0_8qubits),
    (5, 1, qft_no_swap_inverse_approx1_5qubits),
    (6, 1, qft_no_swap_inverse_approx1_6qubits),
    (7, 1, qft_no_swap_inverse_approx1_7qubits),
    (8, 1, qft_no_swap_inverse_approx1_8qubits),
    (5, 2, qft_no_swap_inverse_approx2_5qubits),
    (6, 2, qft_no_swap_inverse_approx2_6qubits),
    (7, 2, qft_no_swap_inverse_approx2_7qubits),
    (8, 2, qft_no_swap_inverse_approx2_8qubits),
    (5, 3, qft_no_swap_inverse_approx3_5qubits),
    (6, 3, qft_no_swap_inverse_approx3_6qubits),
    (7, 3, qft_no_swap_inverse_approx3_7qubits),
    (8, 3, qft_no_swap_inverse_approx3_8qubits),
])
def test_qft_no_swap_inverse(
        circuit_framework: Type[Circuit],
        num_qubits: int,
        approximation_degree: int,
        expected_unitary: NDArray[np.complex128]
    ) -> None:
    """ Test the quantum Fourier transform without swaps and with inverse.
    """
    qft_circuit = circuit_framework(num_qubits)
    qft_circuit.QFT(
        range(num_qubits),
        approximation_degree=approximation_degree,
        do_swaps=False,
        inverse=True
    )
    assert_almost_equal(qft_circuit.get_unitary(), expected_unitary, 8)

@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
@pytest.mark.parametrize("num_qubits, approximation_degree, expected_unitary", [
    (5, 0, qft_swap_inverse_approx0_5qubits),
    (6, 0, qft_swap_inverse_approx0_6qubits),
    (7, 0, qft_swap_inverse_approx0_7qubits),
    (8, 0, qft_swap_inverse_approx0_8qubits),
    (5, 1, qft_swap_inverse_approx1_5qubits),
    (6, 1, qft_swap_inverse_approx1_6qubits),
    (7, 1, qft_swap_inverse_approx1_7qubits),
    (8, 1, qft_swap_inverse_approx1_8qubits),
    (5, 2, qft_swap_inverse_approx2_5qubits),
    (6, 2, qft_swap_inverse_approx2_6qubits),
    (7, 2, qft_swap_inverse_approx2_7qubits),
    (8, 2, qft_swap_inverse_approx2_8qubits),
    (5, 3, qft_swap_inverse_approx3_5qubits),
    (6, 3, qft_swap_inverse_approx3_6qubits),
    (7, 3, qft_swap_inverse_approx3_7qubits),
    (8, 3, qft_swap_inverse_approx3_8qubits),
])
def test_qft_swap_inverse(
        circuit_framework: Type[Circuit],
        num_qubits: int,
        approximation_degree: int,
        expected_unitary: NDArray[np.complex128]
    ) -> None:
    """ Test the quantum Fourier transform with swaps and with inverse.
    """
    qft_circuit = circuit_framework(num_qubits)
    qft_circuit.QFT(
        range(num_qubits),
        approximation_degree=approximation_degree,
        do_swaps=True,
        inverse=True
    )
    assert_almost_equal(qft_circuit.get_unitary(), expected_unitary, 8)