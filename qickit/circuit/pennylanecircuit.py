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

__all__ = ["PennylaneCircuit"]

import copy
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

import pennylane as qml # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit, QiskitCircuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation
from qickit.types import Collection


class PennylaneCircuit(Circuit):
    """ `qickit.circuit.PennylaneCircuit` is the wrapper for using Xanadu's PennyLane in Qickit SDK.
    ref: https://arxiv.org/pdf/1811.04968

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `circuit` : list[qml.Operation]
        The circuit.
    `device` : qml.Device
        The PennyLane device to use.
    `measured_qubits` : list[bool]
        The measurement status of the qubits.
    `circuit_log` : list[dict]
        The circuit log.

    Raises
    ------
    TypeError
        Number of qubits bits must be integers.
    ValueError
        Number of qubits bits must be greater than 0.

    Usage
    -----
    >>> circuit = PennylaneCircuit(num_qubits=2)
    """
    def __init__(self,
                 num_qubits: int) -> None:
        super().__init__(num_qubits=num_qubits)

        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit: list = []

    def _single_qubit_gate(self,
                           gate: Literal["I", "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"],
                           qubit_indices: int | Collection[int],
                           angle: float=0) -> None:
        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "I": qml.Identity(0).matrix(),
            "X": qml.PauliX(0).matrix(),
            "Y": qml.PauliY(0).matrix(),
            "Z": qml.PauliZ(0).matrix(),
            "H": qml.Hadamard(0).matrix(),
            "S": qml.S(0).matrix(),
            "T": qml.T(0).matrix(),
            "RX": qml.RX(angle, wires=0).matrix(),
            "RY": qml.RY(angle, wires=0).matrix(),
            "RZ": qml.RZ(angle, wires=0).matrix()
        }

        # Apply the gate to the specified qubit(s)
        if isinstance(qubit_indices, Collection):
            for index in qubit_indices:
                self.circuit.append(qml.QubitUnitary(gate_mapping[gate], wires=index))
        else:
            self.circuit.append(qml.QubitUnitary(gate_mapping[gate], wires=qubit_indices))

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # Create a single qubit unitary gate
        u3 = qml.U3

        self.circuit.append(u3(theta=angles[0], phi=angles[1], delta=angles[2], wires=qubit_index))

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        # Create a SWAP gate
        swap = qml.SWAP

        self.circuit.append(swap(wires=[first_qubit, second_qubit]))

    def _controlled_qubit_gate(self,
                               gate: Literal["I", "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"],
                               control_indices: int | Collection[int],
                               target_indices: int | Collection[int],
                               angle: float=0) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": qml.PauliX(0).matrix(),
            "Y": qml.PauliY(0).matrix(),
            "Z": qml.PauliZ(0).matrix(),
            "H": qml.Hadamard(0).matrix(),
            "S": qml.S(0).matrix(),
            "T": qml.T(0).matrix(),
            "RX": qml.RX(angle, wires=0).matrix(),
            "RY": qml.RY(angle, wires=0).matrix(),
            "RZ": qml.RZ(angle, wires=0).matrix(),
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    gate_mapping[gate],
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.U3(theta=angles[0], phi=angles[1], delta=angles[2], wires=0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCSWAP(self,
               control_indices: int | Collection[int],
               first_target_index: int,
               second_target_index: int) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        self.circuit.append(
            qml.ControlledQubitUnitary(
                qml.SWAP(wires=[0, 1]).matrix(),
                control_wires=control_indices,
                wires=[first_target_index, second_target_index]
            )
        )

    @Circuit.gatemethod
    def GlobalPhase(self,
                    angle: float) -> None:
        # Create a Global Phase gate
        global_phase = qml.GlobalPhase

        self.circuit.append(global_phase(-angle))

    @Circuit.gatemethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
        # In PennyLane, we apply measurements in '.get_statevector', and '.get_counts'
        # methods. This is due to the need for PennyLane quantum functions to return measurement results.
        # Therefore, we do not need to do anything here.
        if isinstance(qubit_indices, int):
            self.measured_qubits[qubit_indices] = True
        else:
            list(map(lambda qubit_index: self.measured_qubits.__setitem__(qubit_index, True), qubit_indices))

    def get_statevector(self,
                        backend: Backend | None = None) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = copy.deepcopy(self)

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        def compile() -> qml.StateMP:
            """ Compile the circuit.

            Parameters
            ----------
            circuit : Collection[qml.Op]
                The list of operations representing the circuit.

            Returns
            -------
            qml.StateMP
                The state vector of the circuit.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                qml.apply(op)

            return qml.state()

        if backend is None:
            state_vector = qml.QNode(compile, circuit.device)()
        else:
            state_vector = backend.get_statevector(circuit)

        # Round off the small values to 0 (values below 1e-12 are set to 0)
        state_vector = np.round(state_vector, 12)

        # Create masks for real and imaginary parts
        real_mask = (state_vector.imag == 0)
        imag_mask = (state_vector.real == 0)

        # Calculate the sign for each part
        real_sign = np.sign(state_vector.real) * real_mask
        imag_sign = np.sign(state_vector.imag) * imag_mask

        # Calculate the sign for complex numbers
        complex_sign = np.sign(state_vector.real * (np.abs(state_vector.real) <= np.abs(state_vector.imag)) + \
                               state_vector.imag * (np.abs(state_vector.imag) < np.abs(state_vector.real))) * \
                               ~(real_mask | imag_mask)

        # Define the signs for the real and imaginary components
        signs = real_sign + imag_sign + complex_sign

        # Multiply the state vector by the signs
        state_vector = signs * np.abs(state_vector)

        return np.array(state_vector)

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict[str, int]:
        if not(any(self.measured_qubits)):
            raise ValueError("At least one qubit must be measured.")

        # Set the seed
        np.random.seed(0)

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = copy.deepcopy(self)

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Extract what qubits are measured
        qubits_to_measure = [i for i in range(circuit.num_qubits) if circuit.measured_qubits[i]]

        def compile() -> Collection[qml.ProbabilityMP]:
            """ Compile the circuit.

            Parameters
            ----------
            circuit : Collection[qml.Op]
                The list of operations representing the circuit.

            Returns
            -------
            Collection[qml.ProbabilityMP]
                The list of probability measurements.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                qml.apply(op)

            return qml.counts(wires=qubits_to_measure, all_outcomes=True)

        if backend is None:
            device = qml.device(circuit.device.name, wires=circuit.num_qubits, shots=num_shots)
            result = qml.QNode(compile, device)()
            counts = {list(result.keys())[i]: int(list(result.values())[i]) for i in range(len(result))}
        else:
            result = backend.get_counts(self, num_shots=num_shots)

        return counts

    def get_depth(self) -> int:
        circuit = self.convert(QiskitCircuit)
        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = copy.deepcopy(self)

        def compile() -> None:
            """ Compile the circuit.

            Parameters
            ----------
            `circuit` : Collection[qml.Op]
                The list of operations representing the circuit.
            """
            if circuit.circuit == [] or (
                isinstance(circuit.circuit[0], qml.GlobalPhase) and len(circuit.circuit) == 1
            ):
                for i in range(circuit.num_qubits):
                    circuit.circuit.append(qml.Identity(wires=i))

            # Apply the operations in the circuit
            for op in circuit.circuit:
                qml.apply(op)

        # Run the circuit and define the unitary matrix
        unitary = np.array(qml.matrix(compile, wire_order=range(self.num_qubits))(), dtype=complex)

        # PennyLane's `.matrix` function does not take qubit ordering into account,
        # so we need to manually convert the unitary matrix from MSB to LSB
        def MSB_to_LSB(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
            """ Convert the MSB to LSB.

            Parameters
            ----------
            `matrix` : NDArray[np.complex128]
                The matrix to convert.

            Returns
            -------
            `reordered_matrix` : NDArray[np.complex128]
                The new matrix with LSB conversion.
            """
            # Determine the size of the matrix (assuming it's a square matrix)
            size = len(matrix)

            # Create a new matrix to store the reordered elements
            reordered_matrix = np.zeros((size, size), dtype=type(matrix[0][0]))

            # Iterate over each element in the original matrix
            for i in range(size):
                for j in range(size):
                    # Convert the indices from MSB to LSB
                    new_i = int(bin(i)[2:].zfill(int(np.log2(size)))[::-1], 2)
                    new_j = int(bin(j)[2:].zfill(int(np.log2(size)))[::-1], 2)

                    # Assign the value from the original matrix to the new position in the reordered matrix
                    reordered_matrix[new_i][new_j] = matrix[i][j]

            return reordered_matrix

        return MSB_to_LSB(unitary)

    def transpile(self,
                  direct_transpile: bool=True,
                  synthesis_method: UnitaryPreparation | None = None) -> None:
        # Convert to `qickit.circuit.QiskitCircuit` to transpile the circuit
        qiskit_circuit = self.convert(QiskitCircuit)
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.PennylaneCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(PennylaneCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(self,
                qasm_version: int=2) -> str:
        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        pass