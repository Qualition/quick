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

__all__ = ["TestShannonDecomposition"]

from numpy.testing import assert_almost_equal
import pytest
from scipy.stats import unitary_group

from qickit.circuit import QiskitCircuit
from qickit.primitives import Operator
from qickit.synthesis.unitarypreparation import ShannonDecomposition
from tests.synthesis.unitarypreparation import UnitaryPreparationTemplate

# Define the test data
unitary_matrix = unitary_group.rvs(8).astype(complex)


class TestShannonDecomposition(UnitaryPreparationTemplate):
    """ `tests.synthesis.test_shannon_decomposition.TestShannonDecomposition` is the tester class
    for `qickit.synthesis.unitarypreparation.ShannonDecomposition` class.
    """
    def test_init(self) -> None:
        ShannonDecomposition(QiskitCircuit)

    def test_init_invalid_output_framework(self) -> None:
        with pytest.raises(TypeError):
            ShannonDecomposition("invalid_framework") # type: ignore

    def test_prepare_unitary_ndarray(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Prepare the unitary matrix
        circuit = shannon_decomposition.prepare_unitary(unitary_matrix)

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_prepare_unitary_operator(self) -> None:
        # Initialize the Qiskit transpiler
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Prepare the unitary matrix
        circuit = shannon_decomposition.prepare_unitary(Operator(unitary_matrix))

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_a1_a2_optimization(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Generate a 3 and 4 qubit unitary matrix
        unitary_matrix_3 = unitary_group.rvs(8).astype(complex)
        unitary_matrix_4 = unitary_group.rvs(16).astype(complex)

        # Prepare the unitary matrix
        circuit_3 = shannon_decomposition.prepare_unitary(unitary_matrix_3)
        circuit_4 = shannon_decomposition.prepare_unitary(unitary_matrix_4)

        # The number of CX gates in the 3 and 4 qubit circuits must be 20 and 100 respectively
        circuit_3.transpile()
        circuit_4.transpile()

        circuit_3_counts = circuit_3.count_ops()
        circuit_4_counts = circuit_4.count_ops()

        # MCX here is the same as CX gate
        # This is due to the `qickit.circuit.from_framework.FromQiskit` module
        # interpreting all controlled X gates as MCX
        assert circuit_3_counts["CX"] <= 20
        assert circuit_4_counts["CX"] <= 100

    def test_apply_unitary_ndarray(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Initialize the qickit circuit
        circuit = QiskitCircuit(3)

        # Apply the unitary matrix to the circuit
        circuit = shannon_decomposition.apply_unitary(circuit, unitary_matrix, range(3))

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_apply_unitary_operator(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Initialize the qickit circuit
        circuit = QiskitCircuit(3)

        # Apply the unitary matrix to the circuit
        circuit = shannon_decomposition.apply_unitary(circuit, Operator(unitary_matrix), range(3))

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_apply_unitary_invalid_input(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Initialize the qickit circuit
        circuit = QiskitCircuit(3)

        with pytest.raises(TypeError):
            shannon_decomposition.apply_unitary(circuit, "invalid_input", range(3)) # type: ignore

    def test_apply_unitary_invalid_qubit_indices(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Initialize the qickit circuit
        circuit = QiskitCircuit(3)

        with pytest.raises(TypeError):
            shannon_decomposition.apply_unitary(circuit, unitary_matrix, "invalid_indices") # type: ignore

        with pytest.raises(TypeError):
            shannon_decomposition.apply_unitary(circuit, unitary_matrix, [1+1j, 2+2j, 3+3j]) # type: ignore

        with pytest.raises(ValueError):
            shannon_decomposition.apply_unitary(circuit, unitary_matrix, [0, 1, 2, 3])

    def test_apply_unitary_invalid_qubit_indices_out_of_range(self) -> None:
        # Initialize the Shannon decomposition
        shannon_decomposition = ShannonDecomposition(QiskitCircuit)

        # Initialize the qickit circuit
        circuit = QiskitCircuit(3)

        with pytest.raises(IndexError):
            shannon_decomposition.apply_unitary(circuit, unitary_matrix, [0, 1, 4])