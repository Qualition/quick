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

__all__ = ['TKETCircuit']

import copy
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

# TKET imports
from pytket import Circuit as TKCircuit
from pytket import OpType
from pytket.circuit import Op, QControlBox
from pytket.extensions.qiskit import AerBackend

# Qiskit imports
import qiskit # type: ignore
from qiskit import transpile # type: ignore

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.Backend`
if TYPE_CHECKING:
    from qickit.backend import Backend

# Import `qickit.types.collection.Collection`
from qickit.types import Collection


class TKETCircuit(Circuit):
    """ `qickit.circuit.TKETCircuit` is the wrapper for using Quantinuum's TKET in Qickit SDK.

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
    `circuit` : pytket.Circuit
        The TKET circuit.
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

        # Define the circuit
        self.circuit = TKCircuit(self.num_qubits, self.num_clbits)

    @Circuit.gatemethod
    def X(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-X gate
        x = OpType.X

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.add_gate(x, [index])
        else:
            # If it's not a list, apply the X gate to the single qubit
            self.circuit.add_gate(x, [qubit_indices])

    @Circuit.gatemethod
    def Y(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Y gate
        y = OpType.Y

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.add_gate(y, [index])
        else:
            # If it's not a list, apply the Y gate to the single qubit
            self.circuit.add_gate(y, [qubit_indices])

    @Circuit.gatemethod
    def Z(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Z gate
        z = OpType.Z

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.add_gate(z, [index])
        else:
            # If it's not a list, apply the Z gate to the single qubit
            self.circuit.add_gate(z, [qubit_indices])

    @Circuit.gatemethod
    def H(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Hadamard gate
        h = OpType.H

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.add_gate(h, [index])
        else:
            # If it's not an list, apply the H gate to the single qubit
            self.circuit.add_gate(h, [qubit_indices])

    @Circuit.gatemethod
    def S(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-S gate
        s = OpType.S

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.add_gate(s, [index])
        else:
            # If it's not a list, apply the S gate to the single qubit
            self.circuit.add_gate(s, [qubit_indices])

    @Circuit.gatemethod
    def T(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-T gate
        t = OpType.T

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.add_gate(t, [index])
        else:
            # If it's not a list, apply the T gate to the single qubit
            self.circuit.add_gate(t, [qubit_indices])

    @Circuit.gatemethod
    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RX gate with the specified angle
        rx = OpType.Rx
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.add_gate(rx, angle/np.pi, [qubit_index])

    @Circuit.gatemethod
    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RY gate with the specified angle
        ry = OpType.Ry
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.add_gate(ry, angle/np.pi, [qubit_index])

    @Circuit.gatemethod
    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RZ gate with the specified angle
        rz = OpType.Rz
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.add_gate(rz, angle/np.pi, [qubit_index])

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # Create a single qubit unitary gate
        u3 = OpType.U3
        # Apply the U3 gate to the circuit at the specified qubit
        self.circuit.add_gate(u3, [angles[i]/np.pi for i in range(3)], [qubit_index])

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        # Create a SWAP gate
        swap = OpType.SWAP
        # Apply the SWAP gate to the circuit at the specified qubits
        self.circuit.add_gate(swap, [first_qubit, second_qubit])

    @Circuit.gatemethod
    def CX(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-X gate
        cx = OpType.CX
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cx, [control_index, target_index])

    @Circuit.gatemethod
    def CY(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Y gate
        cy = OpType.CY
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cy, [control_index, target_index])

    @Circuit.gatemethod
    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Z gate
        cz = OpType.CZ
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cz, [control_index, target_index])

    @Circuit.gatemethod
    def CH(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-H gate
        ch = OpType.CH
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(ch, [control_index, target_index])

    @Circuit.gatemethod
    def CS(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-S gate
        cs = OpType.CS
        # Apply the CS gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cs, [control_index, target_index])

    @Circuit.gatemethod
    def CT(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-T gate
        t = Op.create(OpType.T)
        ct = QControlBox(t, 1)
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.add_qcontrolbox(ct, [control_index, target_index])

    @Circuit.gatemethod
    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RX gate with the specified angle
        crx = OpType.CRx
        # Apply the CRX gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(crx, angle/np.pi, [control_index, target_index])

    @Circuit.gatemethod
    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RY gate with the specified angle
        cry = OpType.CRy
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cry, angle/np.pi, [control_index, target_index])

    @Circuit.gatemethod
    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RZ gate with the specified angle
        crz = OpType.CRz
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(crz, angle/np.pi, [control_index, target_index])

    @Circuit.gatemethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-U3 gate with the specified angles
        cu3 = OpType.CU3
        # Apply the CU3 gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cu3, [angles[i]/np.pi for i in range(3)], [control_index, target_index])

    @Circuit.gatemethod
    def CSWAP(self,
              control_index: int,
              first_target_index: int,
              second_target_index: int) -> None:
        # Create a Controlled-SWAP gate
        cswap = OpType.CSWAP
        # Apply the CSWAP gate to the circuit at the specified control and target qubits
        self.circuit.add_gate(cswap, [control_index, first_target_index, second_target_index])

    @Circuit.gatemethod
    def MCX(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled X gate with the number of control qubits equal to
        # the length of control_indices
        mcx = OpType.CnX

        # Apply the MCX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_gate(mcx, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCY(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled Y gate with the number of control qubits equal to
        # the length of control_indices
        mcy = OpType.CnY

        # Apply the MCY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_gate(mcy, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCZ(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled Z gate with the number of control qubits equal to
        # the length of control_indices
        mcz = OpType.CnZ

        # Apply the MCZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_gate(mcz, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCH(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled H gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        h = Op.create(OpType.H)
        mch = QControlBox(h, len(control_indices))

        # Apply the MCH gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mch, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCS(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled S gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        s = Op.create(OpType.S)
        mcs = QControlBox(s, len(control_indices))

        # Apply the MCS gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcs, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCT(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled T gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        t = Op.create(OpType.T)
        mct = QControlBox(t, len(control_indices))

        # Apply the MCT gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mct, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCRX(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled RX gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        rx = Op.create(OpType.Rx, angle/np.pi)
        mcrx = QControlBox(rx, len(control_indices))

        # Apply the MCRX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcrx, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCRY(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled RY gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        ry = Op.create(OpType.Ry, angle/np.pi)
        mcry = QControlBox(ry, len(control_indices))

        # Apply the MCRY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcry, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCRZ(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled RZ gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        rz = Op.create(OpType.Rz, angle/np.pi)
        mcrz = QControlBox(rz, len(control_indices))

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcrz, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        u3 = Op.create(OpType.U3, [angles[i]/np.pi for i in range(3)])
        mcu3 = QControlBox(u3, len(control_indices))

        # Apply the MCU3 gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcu3, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCSWAP(self,
               control_indices: int | Collection[int],
               first_target_index: int,
               second_target_index: int) -> None:
        # Ensure control_indices is always treated as a list
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of control_indices
        swap = Op.create(OpType.SWAP)
        mcswap = QControlBox(swap, len(control_indices))

        # Apply the MCSWAP gate to the circuit at the control and target qubits
        self.circuit.add_gate(mcswap, control_indices[:] + [first_target_index, second_target_index])

    @Circuit.gatemethod
    def GlobalPhase(self,
                    angle: float) -> None:
        # Create a Global Phase gate, and apply it to the circuit
        self.circuit.add_phase(angle/np.pi)

    @Circuit.gatemethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
        # Measure the qubits
        if isinstance(qubit_indices, int):
            self.circuit.Measure(qubit_indices, qubit_indices)

        elif isinstance(qubit_indices, Collection):
            for index in qubit_indices:
                self.circuit.Measure(index, index)

        # Set the measurement as applied
        self.measured = True

    def get_statevector(self,
                        backend: Backend | None=None) -> Collection[float]:
        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = copy.deepcopy(self)

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            # Run the circuit and define the state vector
            state_vector = circuit.circuit.get_statevector()

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

        # Return the state vector
        return state_vector

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None=None) -> dict[str, int]:
        if self.measured is False:
            self.measure(range(self.num_qubits))

        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = copy.deepcopy(self)

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            # If no backend is provided, use the AerBackend
            base_backend: AerBackend = AerBackend()
            # Run the circuit
            result = base_backend.get_result(base_backend.process_circuit(circuit.circuit, n_shots=num_shots, seed=0))
            # Get the counts
            counts = {''.join(map(str, basis_state)): num_counts
                      for basis_state, num_counts in result.get_counts().items()}

        else:
            # Run the circuit on the specified backend
            counts = backend.get_counts(self.circuit, num_shots=num_shots)

        # Return the counts
        return counts

    def get_depth(self) -> int:
        # Convert the circuit to Qiskit
        circuit = self.convert(QiskitCircuit)

        # Return the effective depth of the circuit (the number of U3 and CX operations)
        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.number]:
        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = copy.deepcopy(self)

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Run the circuit and define the unitary matrix
        unitary = circuit.circuit.get_unitary()

        # Return the unitary matrix
        return np.array(unitary)

    def transpile(self) -> None:
        # Convert the circuit to QiskitCircuit
        circuit = self.convert(QiskitCircuit)

        # Use the built-in transpiler from IBM Qiskit to transpile the circuit
        transpiled_circuit: qiskit.QuantumCircuit = transpile(circuit.circuit,
                                                              basis_gates = ['cx', 'u3'],
                                                              optimization_level=3,
                                                              seed_transpiler=0)

        # Reset the circuit log (as we will be creating a new one given the transpiled circuit)
        self.reset()

        # Iterate over the gates in the transpiled circuit
        for gate in transpiled_circuit.data:
            # Add the U3 gate to circuit log
            if gate[0].name == 'u3':
                self.U3(gate[0].params, gate[1][0]._index)

            # Add the CX gate to circuit log
            else:
                self.CX(gate[1][0]._index, gate[1][1]._index)

        # Update the global phase
        self.GlobalPhase(transpiled_circuit.global_phase)

    def to_qasm(self) -> str:
        # Convert the circuit to QASM
        qasm = self.convert(QiskitCircuit).circuit.qasm()

        # Return the QASM
        return qasm

    def draw(self) -> None:
        pass