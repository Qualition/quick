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

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

import pennylane as qml # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit, QiskitCircuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation


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
    `measured_qubits` : set[int]
        The set of measured qubits indices.
    `circuit_log` : list[dict]
        The circuit log.
    `process_gate_params_flag` : bool
        The flag to process the gate parameters.

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
    def __init__(
            self,
            num_qubits: int
        ) -> None:

        super().__init__(num_qubits=num_qubits)

        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit: list[qml.Operation] = []

    def _single_qubit_gate(
            self,
            gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
            qubit_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the single qubit gates
        gate_mapping = {
            "I": lambda: qml.Identity(0).matrix(),
            "X": lambda: qml.PauliX(0).matrix(),
            "Y": lambda: qml.PauliY(0).matrix(),
            "Z": lambda: qml.PauliZ(0).matrix(),
            "H": lambda: qml.Hadamard(wires=0).matrix(),
            "S": lambda: qml.S(wires=0).matrix(),
            "Sdg": lambda: qml.adjoint(qml.S(0)).matrix(), # type: ignore
            "T": lambda: qml.T(wires=0).matrix(),
            "Tdg": lambda: qml.adjoint(qml.T(0)).matrix(), # type: ignore
            "RX": lambda: qml.RX(phi=angle, wires=0).matrix(), # type: ignore
            "RY": lambda: qml.RY(phi=angle, wires=0).matrix(), # type: ignore
            "RZ": lambda: qml.RZ(phi=angle, wires=0).matrix() # type: ignore
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        single_qubit_gate = gate_mapping[gate]()

        # Apply the single qubit gate to each qubit index
        for index in qubit_indices:
            self.circuit.append(qml.QubitUnitary(single_qubit_gate, wires=index))

    def U3(
            self,
            angles: Sequence[float],
            qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.U3.__name__, params=locals())

        # Create a single qubit unitary gate
        u3 = qml.U3
        self.circuit.append(u3(theta=angles[0], phi=angles[1], delta=angles[2], wires=qubit_index)) # type: ignore

    def SWAP(
            self,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.SWAP.__name__, params=locals())

        # Create a SWAP gate
        swap = qml.SWAP
        self.circuit.append(swap(wires=[first_qubit_index, second_qubit_index]))

    def _controlled_qubit_gate(
            self,
            gate: Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": lambda: qml.PauliX(0).matrix(),
            "Y": lambda: qml.PauliY(0).matrix(),
            "Z": lambda: qml.PauliZ(0).matrix(),
            "H": lambda: qml.Hadamard(wires=0).matrix(),
            "S": lambda: qml.S(wires=0).matrix(),
            "Sdg": lambda: qml.adjoint(qml.S(0)).matrix(), # type: ignore
            "T": lambda: qml.T(wires=0).matrix(),
            "Tdg": lambda: qml.adjoint(qml.T(0)).matrix(), # type: ignore
            "RX": lambda: qml.RX(phi=angle, wires=0).matrix(), # type: ignore
            "RY": lambda: qml.RY(phi=angle, wires=0).matrix(), # type: ignore
            "RZ": lambda: qml.RZ(phi=angle, wires=0).matrix(), # type: ignore
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        controlled_qubit_gate = gate_mapping[gate]()

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    controlled_qubit_gate,
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    def MCU3(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.MCU3.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.U3(theta=angles[0], phi=angles[1], delta=angles[2], wires=0).matrix(), # type: ignore
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    def MCSWAP(
            self,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:

        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        self.circuit.append(
            qml.ControlledQubitUnitary(
                qml.SWAP(wires=[0, 1]).matrix(),
                control_wires=control_indices,
                wires=[first_target_index, second_target_index]
            )
        )

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate
        global_phase = qml.GlobalPhase
        self.circuit.append(global_phase(-angle))

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        # NOTE: In PennyLane, we apply measurements in '.get_statevector', and '.get_counts'
        # methods. This is due to the need for PennyLane quantum functions to return measurement results.
        # Therefore, we do not need to do anything here.
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Check if any of the qubits have already been measured
        if any(qubit_index in self.measured_qubits for qubit_index in qubit_indices):
            raise ValueError("The qubit(s) have already been measured.")

        # Set the measurement as applied
        for qubit_index in qubit_indices:
            self.measured_qubits.add(qubit_index)

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy() # type: ignore

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

        return np.array(state_vector)

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:

        np.random.seed(0)

        if len(self.measured_qubits) == 0:
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy() # type: ignore

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        def compile() -> qml.CountsMp:
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

            return qml.counts(wires=circuit.measured_qubits, all_outcomes=True)

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
        circuit: PennylaneCircuit = self.copy() # type: ignore

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
        unitary = np.array(qml.matrix(compile, wire_order=range(self.num_qubits))(), dtype=complex) # type: ignore

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

    def transpile(
            self,
            direct_transpile: bool=True,
            synthesis_method: UnitaryPreparation | None = None
        ) -> None:

        # Convert to `qickit.circuit.QiskitCircuit` to transpile the circuit
        qiskit_circuit = self.convert(QiskitCircuit)
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.PennylaneCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(PennylaneCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        pass