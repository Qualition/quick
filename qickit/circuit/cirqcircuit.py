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

__all__ = ["CirqCircuit"]

import copy
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

# Cirq imports
import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, SWAP, I

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.Backend`
if TYPE_CHECKING:
    from qickit.backend import Backend

# Import `qickit.types.collection.Collection`
from qickit.types import Collection


class CirqCircuit(Circuit):
    """ `qickit.circuit.CirqCircuit` is the wrapper for using Google's Cirq in Qickit SDK.
    ref: https://zenodo.org/records/11398048

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
    `qr` : cirq.LineQubit
        The quantum bit register.
    `circuit` : cirq.Circuit
        The circuit.
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

        # Define the quantum bit register
        self.qr = cirq.LineQubit.range(self.num_qubits)

        # Define the circuit (Need to add an identity, otherwise `.get_unitary()`
        # returns the state instead of the operator of the circuit)
        self.circuit = cirq.Circuit()
        self.circuit.append(I(self.qr[0]))

    @Circuit.gatemethod
    def Identity(self,
                 qubit_indices: int | Collection[int]) -> None:
        # Create an Identity gate
        identity = I

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Identity gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(identity(self.qr[index]))
        else:
            # If it's not a list, apply the Identity gate to the single qubit
            self.circuit.append(identity(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def X(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-X gate
        x = X

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(x(self.qr[index]))
        else:
            # If it's not a list, apply the X gate to the single qubit
            self.circuit.append(x(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def Y(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Y gate
        y = Y

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(y(self.qr[index]))
        else:
            # If it's not a list, apply the Y gate to the single qubit
            self.circuit.append(y(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def Z(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Z gate
        z = Z

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(z(self.qr[index]))
        else:
            # If it's not a list, apply the Z gate to the single qubit
            self.circuit.append(z(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def H(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Hadamard gate
        h = H

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(h(self.qr[index]))
        else:
            # If it's not a list, apply the H gate to the single qubit
            self.circuit.append(h(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def S(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-S gate
        s = S

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(s(self.qr[index]))
        else:
            # If it's not a list, apply the S gate to the single qubit
            self.circuit.append(s(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def T(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-T gate
        t = T

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(t(self.qr[index]))
        else:
            # If it's not a list, apply the T gate to the single qubit
            self.circuit.append(t(self.qr[qubit_indices]))

    @Circuit.gatemethod
    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RX gate with the specified angle
        rx = Rx(rads=angle)
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.append(rx(self.qr[qubit_index]))

    @Circuit.gatemethod
    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RY gate with the specified angle
        ry = Ry(rads=angle)
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.append(ry(self.qr[qubit_index]))

    @Circuit.gatemethod
    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RZ gate with the specified angle
        rz = Rz(rads=angle)
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.append(rz(self.qr[qubit_index]))

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # Create a single qubit unitary gate
        u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
              [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
                                                           np.cos(angles[0]/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        # Apply the U3 gate to the circuit at the specified qubit
        self.circuit.append(U3().on(self.qr[qubit_index]))

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        # Create a SWAP gate
        swap = cirq.SWAP
        # Apply the SWAP gate to the circuit at the specified qubits
        self.circuit.append(swap(self.qr[first_qubit], self.qr[second_qubit]))

    @Circuit.gatemethod
    def CX(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-X gate
        cx = cirq.ControlledGate(sub_gate=X, num_controls=1)
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.append(cx(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CY(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Y gate
        cy = cirq.ControlledGate(sub_gate=Y, num_controls=1)
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.append(cy(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Z gate
        cz = cirq.ControlledGate(sub_gate=Z, num_controls=1)
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.append(cz(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CH(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-H gate
        ch = cirq.ControlledGate(sub_gate=H, num_controls=1)
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.append(ch(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CS(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-S gate
        cs = cirq.ControlledGate(sub_gate=S, num_controls=1)
        # Apply the CS gate to the circuit at the specified control and target qubits
        self.circuit.append(cs(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CT(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-T gate
        ct = cirq.ControlledGate(sub_gate=T, num_controls=1)
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.append(ct(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a controlled-RX gate with the specified angle
        crx = cirq.ControlledGate(sub_gate=Rx(rads=angle), num_controls=1)
        # Apply the CRX gate to the circuit at the specified control and target qubits
        self.circuit.append(crx(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a controlled-RY gate with the specified angle
        cry = cirq.ControlledGate(sub_gate=Ry(rads=angle), num_controls=1)
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.append(cry(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a controlled-RZ gate with the specified angle
        crz = cirq.ControlledGate(sub_gate=Rz(rads=angle), num_controls=1)
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.append(crz(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        # Create a single qubit unitary gate
        u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
              [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
                                                           np.cos(angles[0]/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        # Create a Controlled-U3 gate with the specified angles
        cu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=1)
        # Apply the CU3 gate to the circuit at the specified control and target qubits
        self.circuit.append(cu3(self.qr[control_index], self.qr[target_index]))

    @Circuit.gatemethod
    def CSWAP(self,
              control_index: int,
              first_target_index: int,
              second_target_index: int) -> None:
        # Create a Controlled-SWAP gate
        cswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=1)
        # Apply the CSWAP gate to the circuit at the specified control and target qubits
        self.circuit.append(cswap(self.qr[control_index],
                                  self.qr[first_target_index], self.qr[second_target_index]))

    @Circuit.gatemethod
    def MCX(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled X gate with the number of control qubits equal to
        # the length of control_indices
        mcx = cirq.ControlledGate(sub_gate=X, num_controls=len(control_indices))

        # Apply the MCX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcx(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCY(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled Y gate with the number of control qubits equal to
        # the length of control_indices
        mcy = cirq.ControlledGate(sub_gate=Y, num_controls=len(control_indices))

        # Apply the MCY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcy(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCZ(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled Z gate with the number of control qubits equal to
        # the length of control_indices
        mcz = cirq.ControlledGate(sub_gate=Z, num_controls=len(control_indices))

        # Apply the MCZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcz(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCH(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled H gate with the number of control qubits equal to
        # the length of control_indices
        mch = cirq.ControlledGate(sub_gate=H, num_controls=len(control_indices))

        # Apply the MCH gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mch(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCS(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled S gate with the number of control qubits equal to
        # the length of control_indices
        mcs = cirq.ControlledGate(sub_gate=S, num_controls=len(control_indices))

        # Apply the MCS gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcs(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCT(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled T gate with the number of control qubits equal to
        # the length of control_indices
        mct = cirq.ControlledGate(sub_gate=T, num_controls=len(control_indices))

        # Apply the MCT gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mct(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

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
        mcrx = cirq.ControlledGate(sub_gate=Rx(rads=angle), num_controls=len(control_indices))

        # Apply the MCRX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcrx(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

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
        mcry = cirq.ControlledGate(sub_gate=Ry(rads=angle), num_controls=len(control_indices))

        # Apply the MCRY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcry(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

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
        mcrz = cirq.ControlledGate(sub_gate=Rz(rads=angle), num_controls=len(control_indices))

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcrz(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a single qubit unitary gate
        u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
              [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
                                                           np.cos(angles[0]/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        mcu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=len(control_indices))

        # Apply the MCU3 gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(
                mcu3(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    @Circuit.gatemethod
    def MCSWAP(self,
               control_indices: int | Collection[int],
               first_target_index: int,
               second_target_index: int) -> None:
        # Ensure control_indices is always treated as a list
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of control_indices
        mcswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=len(control_indices))

        # Apply the MCSWAP gate to the circuit at the control and target qubits
        self.circuit.append(
            mcswap(*map(self.qr.__getitem__, control_indices),
                   self.qr[first_target_index], self.qr[second_target_index])
        )

    @Circuit.gatemethod
    def GlobalPhase(self,
                    angle: float) -> None:
        # Create a Global Phase gate (Cirq takes in e^i*angle as the argument)
        global_phase = cirq.GlobalPhaseGate(np.exp(1j*angle))

        # Apply the Global Phase gate to the circuit
        self.circuit.append(global_phase())

    @Circuit.gatemethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Measure the qubits
        self.circuit.append(cirq.measure(
            *map(self.qr.__getitem__, qubit_indices), key="meas"
            )
        )

        # Set the measurement as applied
        self.measured = True

    def get_statevector(self,
                        backend: Backend | None = None) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            # Define the state vector
            state_vector = circuit.circuit.final_state_vector(qubit_order=self.qr)

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

        return np.array(state_vector)

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict:
        # TODO: Add a native sampler
        if backend is None:
            # Run the circuit to get the state vector
            state_vector = self.get_statevector()
            # Format the state vector to get the counts
            counts = {format(int(index),"0{}b".format(self.num_qubits)): int(abs(amplitude)**2 * num_shots) \
                      for index, amplitude in enumerate(state_vector)}

        else:
            # Copy the circuit as the operations are applied inplace
            circuit: CirqCircuit = copy.deepcopy(self)
            # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
            circuit.vertical_reverse()
            # Run the circuit on the specified backend
            counts = backend.get_counts(circuit, num_shots)

        return counts

    def get_depth(self) -> int:
        # Convert the circuit to Qiskit
        circuit = self.convert(QiskitCircuit)

        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.number]:
        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the unitary matrix
        unitary = cirq.unitary(circuit.circuit)

        return np.array(unitary)

    def to_qasm(self,
                qasm_version: int=2) -> str:
        # Convert the circuit to QASM
        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        print(self.circuit)