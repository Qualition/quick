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
    "test_one_qubit_zyz_decomposition",
    "test_one_qubit_u3_decomposition",
    "test_invalid_basis_fail",
    "test_invalid_indices_fail"
]

from numpy.testing import assert_almost_equal
import pytest
from scipy.stats import unitary_group

from qickit.circuit import QiskitCircuit
from qickit.synthesis.gate_decompositions.one_qubit_decomposition import OneQubitDecomposition


def test_one_qubit_zyz_decomposition() -> None:
    """ Test the one qubit ZYZ decomposition.
    """
    # Generate a random unitary matrix
    unitary_matrix = unitary_group.rvs(2).astype(complex)

    # Define a circuit
    circuit = QiskitCircuit(1)

    # Create a one qubit decomposition object
    one_qubit_decomposition = OneQubitDecomposition(output_framework=QiskitCircuit, basis="zyz")

    # Apply the one qubit ZYZ decomposition
    one_qubit_decomposition.apply_unitary(circuit, unitary_matrix, 0)

    # Check that the circuit is equivalent to the original unitary matrix
    assert_almost_equal(circuit.get_unitary(), unitary_matrix, decimal=8)

def test_one_qubit_u3_decomposition() -> None:
    """ Test the one qubit U3 decomposition.
    """
    # Generate a random unitary matrix
    unitary_matrix = unitary_group.rvs(2).astype(complex)

    # Define a circuit
    circuit = QiskitCircuit(1)

    # Create a one qubit decomposition object
    one_qubit_decomposition = OneQubitDecomposition(output_framework=QiskitCircuit, basis="u3")

    # Apply the one qubit U3 decomposition
    one_qubit_decomposition.apply_unitary(circuit, unitary_matrix, 0)

    # Check that the circuit is equivalent to the original unitary matrix
    assert_almost_equal(circuit.get_unitary(), unitary_matrix, decimal=8)

def test_invalid_basis_fail() -> None:
    """ Test that an invalid basis raises an error.
    """
    # Check that an invalid basis raises an error
    with pytest.raises(ValueError):
        OneQubitDecomposition(QiskitCircuit, "invalid_basis") # type: ignore

def test_invalid_indices_fail() -> None:
    """ Test that an invalid number of indices raises an error.
    """
    # Define a circuit
    circuit = QiskitCircuit(1)

    # Create a one qubit decomposition object
    one_qubit_decomposition = OneQubitDecomposition(output_framework=QiskitCircuit, basis="zyz")

    # Check that an invalid number of indices raises an error
    with pytest.raises(ValueError):
        one_qubit_decomposition.apply_unitary(circuit, unitary_group.rvs(2), [0, 1]) # type: ignore