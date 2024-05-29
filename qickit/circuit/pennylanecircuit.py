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

__all__ = ['PennylaneCircuit']

from typing import TYPE_CHECKING
import copy
import numpy as np
from numpy.typing import NDArray

# Pennylane imports
import pennylane as qml # type: ignore

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.Backend`
if TYPE_CHECKING:
    from qickit.backend import Backend

# Import `qickit.types.collection.Collection`
from qickit.types import Collection


class PennylaneCircuit(Circuit):
    """ `qickit.circuit.PennylaneCircuit` is the wrapper for using Xanadu's PennyLane in Qickit SDK.

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `num_clbits` : int
        Number of classical bits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `num_clbits` : int
        Number of classical bits in the circuit.
    `circuit` : list[qml.Operation]
        The circuit.
    `device` : qml.Device
        The PennyLane device to use.
    `measured` : bool
        The measurement status.
    `circuit_log` : list[dict]
        The circuit log.
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        super().__init__(num_qubits=num_qubits,
                         num_clbits=num_clbits)

        # Define the device
        self.device = qml.device("default.qubit", wires=self.num_qubits)

        # Define the circuit
        self.circuit = []

    @Circuit.gatemethod
    def Identity(self,
                 qubit_indices: int | Collection[int]) -> None:
        # Create an Identity gate
        identity = qml.Identity

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Identity gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(identity(wires=index))
        else:
            # If it's not a list, apply the Identity gate to the single qubit
            self.circuit.append(identity(wires=qubit_indices))

    @Circuit.gatemethod
    def X(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-X gate
        x = qml.PauliX

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(x(wires=index))
        else:
            # If it's not a list, apply the X gate to the single qubit
            self.circuit.append(x(wires=qubit_indices))

    @Circuit.gatemethod
    def Y(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Y gate
        y = qml.PauliY

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(y(wires=index))
        else:
            # If it's not a list, apply the Y gate to the single qubit
            self.circuit.append(y(wires=qubit_indices))

    @Circuit.gatemethod
    def Z(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Z gate
        z = qml.PauliZ

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(z(wires=index))
        else:
            # If it's not a list, apply the Z gate to the single qubit
            self.circuit.append(z(wires=qubit_indices))

    @Circuit.gatemethod
    def H(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Hadamard gate
        h = qml.Hadamard

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(h(wires=index))
        else:
            # If it's not an list, apply the H gate to the single qubit
            self.circuit.append(h(wires=qubit_indices))

    @Circuit.gatemethod
    def S(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-S gate
        s = qml.S

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(s(wires=index))
        else:
            # If it's not a list, apply the S gate to the single qubit
            self.circuit.append(s(wires=qubit_indices))

    @Circuit.gatemethod
    def T(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-T gate
        t = qml.T

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(t(wires=index))
        else:
            # If it's not a list, apply the T gate to the single qubit
            self.circuit.append(t(wires=qubit_indices))

    @Circuit.gatemethod
    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RX gate with the specified angle
        rx = qml.RX
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.append(rx(angle, wires=qubit_index))

    @Circuit.gatemethod
    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RY gate with the specified angle
        ry = qml.RY
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.append(ry(angle, wires=qubit_index))

    @Circuit.gatemethod
    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RZ gate with the specified angle
        rz = qml.RZ
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.append(rz(angle, wires=qubit_index))

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # Create a single qubit unitary gate
        u3 = qml.U3
        # Apply the U3 gate to the circuit at the specified qubit
        self.circuit.append(u3(theta=angles[0], phi=angles[1], delta=angles[2], wires=qubit_index))

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        # Create a SWAP gate
        swap = qml.SWAP
        # Apply the SWAP gate to the circuit at the specified qubits
        self.circuit.append(swap(wires=[first_qubit, second_qubit]))

    @Circuit.gatemethod
    def CX(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-X gate
        cx = qml.CNOT
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.append(cx(wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CY(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Y gate
        cy = qml.CY
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.append(cy(wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Z gate
        cz = qml.CZ
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.append(cz(wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CH(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-H gate
        ch = qml.CH
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.append(ch(wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CS(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-S gate
        cs = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=control_index, wires=target_index)
        # Apply the CS gate to the circuit at the specified control and target qubits
        self.circuit.append(cs)

    @Circuit.gatemethod
    def CT(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-T gate
        ct = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=control_index, wires=target_index)
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.append(ct)

    @Circuit.gatemethod
    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RX gate with the specified angle
        crx = qml.CRX
        # Apply the CRX gate to the circuit at the specified control and target qubits
        self.circuit.append(crx(angle, wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RY gate with the specified angle
        cry = qml.CRY
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.append(cry(angle, wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RZ gate with the specified angle
        crz = qml.CRZ
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.append(crz(angle, wires=[control_index, target_index]))

    @Circuit.gatemethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-U3 gate with the specified angles
        cu3 = qml.U3(theta=angles[0], phi=angles[1], delta=angles[2],wires=0).matrix()
        # Apply the CU3 gate to the circuit at the specified control and target qubits
        self.circuit.append(qml.ControlledQubitUnitary(cu3, control_wires=control_index, wires=target_index))

    @Circuit.gatemethod
    def CSWAP(self,
              control_index: int,
              first_target_index: int,
              second_target_index: int) -> None:
        # Create a Controlled-SWAP gate
        cswap = qml.CSWAP
        # Apply the CSWAP gate to the circuit at the specified control and target qubits
        self.circuit.append(cswap(wires=[control_index, first_target_index, second_target_index]))

    @Circuit.gatemethod
    def MCX(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.PauliX(0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCY(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.PauliY(0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCZ(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.PauliZ(0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCH(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCH gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.Hadamard(0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCS(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCS gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.S(0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCT(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCT gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.T(0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCRX(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.RX(angle, wires=0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCRY(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.RY(angle, wires=0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCRZ(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                qml.ControlledQubitUnitary(
                    qml.RZ(angle, wires=0).matrix(),
                    control_wires=control_indices,
                    wires=target_index
                )
            )

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCU3 gate to the circuit at the control and target qubits
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
        # Ensure control_indices is always treated as a list
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Apply the MCSWAP gate to the circuit at the control and target qubits
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

        # Apply the GlobalPhase gate to the circuit
        self.circuit.append(global_phase(-angle))

    @Circuit.gatemethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
        # In PennyLane, we apply measurements in '.get_statevector', and '.get_counts'
        # methods. This is due to the need for PennyLane quantum functions to return measurement results.
        # Therefore, we do not need to do anything here.
        pass

    def get_statevector(self,
                        backend: Backend | None = None) -> Collection[float]:
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
            # Run the circuit and define the state vector
            state_vector = qml.QNode(compile, circuit.device)()

        else:
            # Run the circuit on the specified backend and define the state vector
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

        return state_vector

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict[str, int]:
        # Set the seed
        np.random.seed(0)

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = copy.deepcopy(self)

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

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

            return qml.counts(all_outcomes=True)

        if backend is None:
            # Define the device
            device = qml.device(circuit.device.name, wires=circuit.num_qubits, shots=num_shots)
            # Apply the operations in the circuit
            result = qml.QNode(compile, device)()
            # Get the counts
            counts = {list(result.keys())[i]: list(result.values())[i] for i in range(len(result))}

        else:
            # Run the circuit on the specified backend
            result = backend.get_counts(circuit, num_shots=num_shots)

        return counts

    def get_depth(self) -> int:
        # Convert the circuit to Qiskit
        circuit = self.convert(QiskitCircuit)

        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.number]:
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
        def MSB_to_LSB(matrix: NDArray[np.number]) -> NDArray[np.number]:
            """ Convert the MSB to LSB.

            Parameters
            ----------
            `matrix` : NDArray[np.number]
                The matrix to convert.

            Returns
            -------
            `reordered_matrix` : NDArray[np.number]
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

    def to_qasm(self) -> str:
        # Convert the circuit to QASM
        qasm = self.convert(QiskitCircuit).circuit.qasm()

        return qasm

    def draw(self) -> None:
        pass