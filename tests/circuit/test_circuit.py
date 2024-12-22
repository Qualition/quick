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

__all__ = ["Template"]

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


# Note that this unit test also covers the `qickit/synthesis/gate_decompositions/multi_controlled_decomposition`
# module as the multi-controlled gates in here would fail if the decomposition is incorrect
class Template(ABC):
    """ `tests.circuit.Template` is the template for creating circuit testers.
    """
    @abstractmethod
    def test_init(self) -> None:
        """ Test the initialization of the circuit.
        """

    @abstractmethod
    def test_num_qubits_value(self) -> None:
        """ Test to see if the error is raised when the number of qubits
        is less than or equal to 0.
        """

    @abstractmethod
    def test_num_qubits_type(self) -> None:
        """ Test to see if the error is raised when the number of qubits
        is not an integer.
        """

    @abstractmethod
    def test_single_qubit_gate_from_range(self) -> None:
        """ Test the single qubit gate when indices are passed as a range instance.
        """

    @abstractmethod
    def test_single_qubit_gate_from_tuple(self) -> None:
        """ Test the single qubit gate when indices are passed as a tuple instance.
        """

    @abstractmethod
    def test_single_qubit_gate_from_ndarray(self) -> None:
        """ Test the single qubit gate when indices are passed as a numpy.ndarray instance.
        """

    @abstractmethod
    def test_Identity(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Identity gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_X(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Pauli-X gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_Y(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Pauli-Y gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_Z(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Pauli-Z gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_H(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Hadamard gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_S(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Clifford-S gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_Sdg(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Clifford-S dagger gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_T(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Clifford-T gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_Tdg(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Clifford-T dagger gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_RX(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the RX gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_RY(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the RY gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_RZ(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the RZ gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_Phase(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Phase gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_XPow(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Pauli-X power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_YPow(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Pauli-Y power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_ZPow(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Pauli-Z power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_RXX(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the RXX gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `first_qubit_index`: int
            The first qubit index.
        `second_qubit_index`: int
            The second qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_RYY(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the RYY gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `first_qubit_index`: int
            The first qubit index.
        `second_qubit_index`: int
            The second qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_RZZ(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the RZZ gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `first_qubit_index`: int
            The first qubit index.
        `second_qubit_index`: int
            The second qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_U3(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angles: tuple[float, float, float],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the U3 gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `qubit_indices`: int | list[int]
            The qubit indices.
        `angles`: tuple[float, float, float]
            The rotation angles.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_SWAP(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the SWAP gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `first_qubit_index`: int
            The first qubit index.
        `second_qubit_index`: int
            The second qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CX(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Pauli-X gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CY(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Pauli-Y gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CZ(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Pauli-Z gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CH(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Hadamard gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CS(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Clifford-S gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CSdg(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Clifford-S dagger gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CT(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Clifford-T gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CTdg(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Clifford-T dagger gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CRX(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled RX gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CRY(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled RY gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CRZ(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled RZ gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CPhase(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Phase gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CXPow(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Pauli-X power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CYPow(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Pauli-Y power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CZPow(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled Pauli-Z power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CRXX(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled RXX gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CRYY(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled RYY gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CRZZ(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled RZZ gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CU3(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angles: tuple[float, float, float],
            expected
        ) -> None:
        """ Test the Controlled U3 gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `target_index`: int
            The target qubit index.
        `angles`: tuple[float, float, float]
            The rotation angles.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_CSWAP(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Controlled SWAP gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_index`: int
            The control qubit index.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCX(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Pauli-X gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCY(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCZ(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCH(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Hadamard gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCS(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Clifford-S gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCSdg(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Clifford-S dagger gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCT(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Clifford-T gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCTdg(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Clifford-T dagger gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCRX(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled RX gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCRY(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled RY gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCRZ(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled RZ gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCPhase(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Phase gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCXPow(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Pauli-X power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCYPow(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Pauli-Y power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCZPow(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled Pauli-Z power gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `power`: float
            The power of the gate.
        `global_shift`: float
            The global shift of the gate.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCRXX(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled RXX gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCRYY(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled RYY gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCRZZ(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled RZZ gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `angle`: float
            The rotation angle.
        `expected`: NDArray[np.complex128]
        """

    @abstractmethod
    def test_MCU3(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angles: tuple[float, float, float],
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled U3 gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `target_indices`: int | list[int]
            The target qubit indices.
        `angles`: tuple[float, float, float]
            The rotation angles.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.
        """

    @abstractmethod
    def test_MCSWAP(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the Multi-Controlled SWAP gate.

        Parameters
        ----------
        `num_qubits`: int
            The number of qubits in the circuit.
        `control_indices`: int | list[int]
            The control qubit indices.
        `first_target_index`: int
            The first target qubit index.
        `second_target_index`: int
            The second target qubit index.
        `expected`: NDArray[np.complex128]
            The expected unitary matrix.k
        """

    @abstractmethod
    def test_GlobalPhase(self) -> None:
        """ Test the Global Phase gate.
        """

    @abstractmethod
    def test_single_measurement(self) -> None:
        """ Test the measurement gate for a single index.
        """

    @abstractmethod
    def test_multiple_measurement(self) -> None:
        """ Test the measurement gate for multiple indices.
        """

    @abstractmethod
    def test_get_statevector(self) -> None:
        """ Test the `circuit.get_statevector()` operation.
        """

    @abstractmethod
    def test_get_unitary(self) -> None:
        """ Test the `circuit.get_unitary()` operation.
        """

    @abstractmethod
    def test_partial_get_counts(self) -> None:
        """ Test the `circuit.get_counts()` operation with only partial measurement.
        """

    @abstractmethod
    def test_get_counts(self) -> None:
        """ Test the `circuit.get_counts()` operation.
        """
