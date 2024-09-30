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

__all__ = ["TestUniformlyControlledGates"]

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from typing import Type

from qickit.circuit import Circuit, CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit
from tests.circuit.gate_utils import (
    UCRX_unitary_matrix, UCRY_unitary_matrix, UCRZ_unitary_matrix, UC_unitary_matrix,
    UC_up_to_diagonal_unitary_matrix, UC_mux_simplified_unitary_matrix,
    UC_mux_simplified_up_to_diagonal_unitary_matrix
)


# The quantum circuit frameworks
CIRCUIT_FRAMEWORKS = [CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit]


class TestUniformlyControlledGates:
    """ `tests.circuit.TestUniformlyControlledGates` is the class for testing uniformly controlled gates.

    These gates are:
    - `circuit.UCRX()`
    - `circuit.UCRY()`
    - `circuit.UCRZ()`
    - `circuit.Diagonal()`
    - `circuit.UC()`
    """
    def UCRX(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `UCRX` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        # Define the quantum circuit
        circuit = circuit_framework(3)

        # Apply the UCRX gate
        circuit.UCRX([1, 0], 2, [np.pi/2, np.pi/3, np.pi/4, np.pi/5])

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UCRX_unitary_matrix, decimal=8)

    def UCRY(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `UCRY` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        # Define the quantum circuit
        circuit = circuit_framework(3)

        # Apply the UCRY gate
        circuit.UCRY([1, 0], 2, [np.pi/2, np.pi/3, np.pi/4, np.pi/5])

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UCRY_unitary_matrix, decimal=8)

    def UCRZ(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `UCRZ` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        # Define the quantum circuit
        circuit = circuit_framework(3)

        # Apply the UCRZ gate
        circuit.UCRZ([1, 0], 2, [np.pi/2, np.pi/3, np.pi/4, np.pi/5])

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UCRZ_unitary_matrix, decimal=8)

    def Diagonal(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        # Define the quantum circuit
        circuit = circuit_framework(3)

        # Apply the Diagonal gate
        diagonal = [1, 1, 1, -1, 1, -1, 1, -1]
        circuit.Diagonal(np.array(diagonal), [0, 1, 2])

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), np.diag(diagonal).astype(complex), decimal=8)

    def UC(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        # Define the quantum circuit
        circuit = circuit_framework(3)

        # Define the list of single-qubit gates (H, X, H, X)
        single_qubit_gates = [
            np.array([
                [0.70710678+0.j, 0.70710678+0.j],
                [0.70710678+0.j, -0.70710678+0.j]
            ]),
            np.array([
                [0.+0.j, 1.+0.j],
                [1.+0.j, 0.+0.j]
            ]),
            np.array([
                [0.70710678+0.j, 0.70710678+0.j],
                [0.70710678+0.j, -0.70710678+0.j]
            ]),
            np.array([
                [0.+0.j, 1.+0.j],
                [1.+0.j, 0.+0.j]
            ])
        ]

        # Apply the UC gate
        circuit.UC([0, 1], 2, single_qubit_gates, up_to_diagonal=True, multiplexor_simplification=False)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UC_up_to_diagonal_unitary_matrix, 8)

        circuit = circuit_framework(3)

        # Apply the UC gate not up to diagonal
        circuit.UC([0, 1], 2, single_qubit_gates, up_to_diagonal=False, multiplexor_simplification=False)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UC_unitary_matrix, 8)

        circuit = circuit_framework(3)

        # Apply the UC gate with multiplexor simplification
        circuit.UC([1, 2], 0, single_qubit_gates, up_to_diagonal=True, multiplexor_simplification=True)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UC_mux_simplified_up_to_diagonal_unitary_matrix, 8)

        circuit = circuit_framework(3)

        # Apply the UC gate with multiplexor simplification
        circuit.UC([1, 2], 0, single_qubit_gates, up_to_diagonal=False, multiplexor_simplification=True)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), UC_mux_simplified_unitary_matrix, 8)

    def test_UCRX(self) -> None:
        """ Test the `UCRX` gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.UCRX(circuit_framework)

    def test_UCRY(self) -> None:
        """ Test the `UCRY` gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.UCRY(circuit_framework)

    def test_UCRZ(self) -> None:
        """ Test the `UCRZ` gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.UCRZ(circuit_framework)

    def test_UCPauliRot_invalid_num_angles(self) -> None:
        """ Test the case when the number of angles is invalid.
        """
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.UCRX([1, 0], 2, [np.pi/2])

    def test_UCPauliRot_invalid_num_controls(self) -> None:
        """ Test the case when the number of control qubits is invalid.
        """
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.UCRX([0], 2, [np.pi/2, np.pi/3, np.pi/4, np.pi/5])

    def test_UCPauliRot_invalid_rot_axis(self) -> None:
        """ Test the case when the rotation axis is invalid.
        """
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.UCPauliRot([1, 0], 2, [np.pi/2, np.pi/3, np.pi/4, np.pi/5], rot_axis="invalid") # type: ignore

    def test_Diagonal(self) -> None:
        """ Test the `Diagonal` gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.Diagonal(circuit_framework)

    def test_Diagonal_invalid_diagonal(self) -> None:
        """ Test the case when the diagonal is invalid.
        """
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.Diagonal(np.array([1, 1, 1, -1, 1, -1, 1]), [0, 1, 2])

        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.Diagonal(np.array([1, 1, 1, -1, 1, -1, 1, 0]), [0, 1, 2])

    def test_Diagonal_invalid_number_of_qubits(self) -> None:
        """ Test the case when the number of qubits is invalid.
        """
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.Diagonal(np.array([1, 1, 1, -1, 1, -1, 1, -1]), [0, 1])

    def test_UC(self) -> None:
        """ Test the `UC` gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.UC(circuit_framework)

    def test_UC_invalid_single_qubit_gate(self) -> None:
        """ Test the case when the single-qubit gates are invalid.
        """
        # Invalid dimension
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.UC([0, 1], 2, [np.array([[1, 0, 0], [0, 0, 1]])])

        # Invalid number of single-qubit gates
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.UC([0, 1], 2, [np.eye(2), np.eye(2), np.eye(2)]) # type: ignore

        # Non-unitary matrix
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(2)
            circuit.UC([0], 1, [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [1, 1]])])

    def test_UC_invalid_number_of_qubits(self) -> None:
        """ Test the case when the number of qubits is invalid.
        """
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(3)
            circuit.UC([0, 1], 2, [np.eye(2), np.eye(2)]) # type: ignore