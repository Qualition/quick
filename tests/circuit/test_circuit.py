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
    def test_Identity(self) -> None:
        """ Test the Identity gate.
        """

    @abstractmethod
    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """

    @abstractmethod
    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """

    @abstractmethod
    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """

    @abstractmethod
    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """

    @abstractmethod
    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """

    @abstractmethod
    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """

    @abstractmethod
    def test_RX(self) -> None:
        """ Test the RX gate.
        """

    @abstractmethod
    def test_RY(self) -> None:
        """ Test the RY gate.
        """

    @abstractmethod
    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """

    @abstractmethod
    def test_U3(self) -> None:
        """ Test the U3 gate.
        """

    @abstractmethod
    def test_SWAP(self) -> None:
        """ Test the SWAP gate.
        """

    @abstractmethod
    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """

    @abstractmethod
    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """

    @abstractmethod
    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """

    @abstractmethod
    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """

    @abstractmethod
    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """

    @abstractmethod
    def test_CT(self) -> None:
        """ Test the Controlled Clifford-T gate.
        """

    @abstractmethod
    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """

    @abstractmethod
    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """

    @abstractmethod
    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """

    @abstractmethod
    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """

    @abstractmethod
    def test_CSWAP(self) -> None:
        """ Test the Controlled SWAP gate.
        """

    @abstractmethod
    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """

    @abstractmethod
    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """

    @abstractmethod
    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """

    @abstractmethod
    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """

    @abstractmethod
    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """

    @abstractmethod
    def test_MCT(self) -> None:
        """ Test the Multi-Controlled Clifford-T gate.
        """

    @abstractmethod
    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """

    @abstractmethod
    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """

    @abstractmethod
    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """

    @abstractmethod
    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """

    @abstractmethod
    def test_MCSWAP(self) -> None:
        """ Test the Multi-Controlled SWAP gate.
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
    def test_remove_measurement(self) -> None:
        """ Test the removal of measurement gate.
        """

    @abstractmethod
    def test_unitary(self) -> None:
        """ Test the unitary gate.
        """

    @abstractmethod
    def test_get_statevector(self) -> None:
        """ Test the get_statevector operation.
        """

    @abstractmethod
    def test_partial_get_counts(self) -> None:
        """ Test the get_counts operation with only partial measurement.
        """

    @abstractmethod
    def test_get_counts(self) -> None:
        """ Test the get_counts operation.
        """

    @abstractmethod
    def test_vertical_reverse(self) -> None:
        """ Test the vertical reverse operation.
        """

    @abstractmethod
    def test_horizontal_reverse(self) -> None:
        """ Test the horizontal reverse operation.
        """

    @abstractmethod
    def test_add(self) -> None:
        """ Test the addition operation.
        """

    @abstractmethod
    def test_add_fail(self) -> None:
        """ Test the failure of the addition operation.
        """

    @abstractmethod
    def test_transpile(self) -> None:
        """ Test the transpile operation.
        """

    @abstractmethod
    def test_get_depth(self) -> None:
        """ Test the get_depth operation.
        """

    @abstractmethod
    def test_get_width(self) -> None:
        """ Test the get_width operation.
        """

    @abstractmethod
    def test_compress(self) -> None:
        """ Test the compress operation.
        """

    @abstractmethod
    def test_compress_fail(self) -> None:
        """ Test the failure of the compress operation.
        """

    @abstractmethod
    def test_change_mapping(self) -> None:
        """ Test the change_mapping operation.
        """

    @abstractmethod
    def test_reset(self) -> None:
        """ Test the reset operation.
        """