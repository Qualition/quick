# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the GNU Version 3.0 (the "License");
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

__all__ = ['CirqCircuit']

from collections.abc import Iterable
import numpy as np
import math

# Cirq imports
import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, CX, CZ
from cirq.circuits.qasm_output import QasmUGate
CY = cirq.ControlledGate(Y)

# Qiskit imports
import qiskit
from qiskit import transpile

# Import `qickit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.Backend`
from qickit.backend import Backend


class CirqCircuit(Circuit):
    """ `CirqCircuit` is the wrapper for using Google's Cirq in Qickit SDK.
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        """ Initialize a `qickit.CirqCircuit` instance.

        Parameters
        ----------
        `num_qubits` (int):
            Number of qubits in the circuit.
        `num_clbits` (int):
            Number of classical bits in the circuit.
        """
        # Define the number of quantum bits
        self.num_qubits = num_qubits
        # Define the number of classical bits
        self.num_clbits = num_clbits

        # Define the quantum bit register
        self.qr = cirq.LineQubit.range(self.num_qubits)

        # Define the circuit
        self.circuit = cirq.Circuit()

        # Define the measurement status
        self.measured = False

        # Define the circuit log (list[dict])
        self.circuit_log = []

    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        """ Apply a RX gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `qubit_index` (int):
            The index of the qubit to apply the gate to.
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the RX gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

         # Create an RX gate with the specified angle
        rx = Rx(rads=angle)
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.append(rx(self.qr[qubit_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'RX', 'angle': angle, 'qubit_index': qubit_index})

    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        """ Apply a RY gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `qubit_index` (int):
            The index of the qubit to apply the gate to.
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the RY gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Create an RY gate with the specified angle
        ry = Ry(rads=angle)
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.append(ry(self.qr[qubit_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'RY', 'angle': angle, 'qubit_index': qubit_index})

    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        """ Apply a RZ gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `qubit_index` (int):
            The index of the qubit to apply the gate to.
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the RZ gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Create an RZ gate with the specified angle
        rz = Rz(rads=angle)
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.append(rz(self.qr[qubit_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'RZ', 'angle': angle, 'qubit_index': qubit_index})

    def H(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Hadamard gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Hadamard gate
        h = H

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(h(self.qr[index]))
        else:
            # If it's not a list, apply the H gate to the single qubit
            self.circuit.append(h(self.qr[qubit_indices]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'H', 'qubit_indices': qubit_indices})

    def X(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Pauli-X gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Pauli X gate
        x = X

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(x(self.qr[index]))
        else:
            # If it's not a list, apply the X gate to the single qubit
            self.circuit.append(x(self.qr[qubit_indices]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'X', 'qubit_indices': qubit_indices})

    def Y(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Pauli-Y gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Pauli Y gate
        y = Y

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(y(self.qr[index]))
        else:
            # If it's not a list, apply the Y gate to the single qubit
            self.circuit.append(y(self.qr[qubit_indices]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'Y', 'qubit_indices': qubit_indices})

    def Z(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Pauli-Z gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Pauli Z gate
        z = Z

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(z(self.qr[index]))
        else:
            # If it's not a list, apply the Z gate to the single qubit
            self.circuit.append(z(self.qr[qubit_indices]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'Z', 'qubit_indices': qubit_indices})

    def S(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a S gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Clifford S gate
        s = S

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(s(self.qr[index]))
        else:
            # If it's not a list, apply the S gate to the single qubit
            self.circuit.append(s(self.qr[qubit_indices]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'S', 'qubit_indices': qubit_indices})

    def T(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a T gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Clifford T gate
        t = T

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(t(self.qr[index]))
        else:
            # If it's not a list, apply the T gate to the single qubit
            self.circuit.append(t(self.qr[qubit_indices]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'T', 'qubit_indices': qubit_indices})

    def U3(self,
           angles: Iterable[float],
           qubit_index: int) -> None:
        """ Apply a U3 gate to the circuit.

        Parameters
        ----------
        `angles` (Iterable[float]):
            The rotation angles in radians.
        `qubit_index` (int):
            The index of the qubit to apply the gate to.
        """
        # Create a single qubit unitary gate
        u3 = QasmUGate(angles[0] / np.pi, angles[1] / np.pi, angles[2] / np.pi)
        # Apply the U3 gate to the circuit at the specified qubit
        self.circuit.append(u3(self.qr[qubit_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'U3', 'angles': angles, 'qubit_index': qubit_index})

    def CX(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a CX gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-X gate
        cx = CX
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.append(cx(self.qr[control_index], self.qr[target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CX', 'control_index': control_index, 'target_index': target_index})

    def CY(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a CY gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-Y gate (Cirq does not have a built-in CY gate)
        cy = CY
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.append(cy(self.qr[control_index], self.qr[target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CY', 'control_index': control_index, 'target_index': target_index})

    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a CZ gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-Z gate
        cz = CZ
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.append(cz(self.qr[control_index], self.qr[target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CZ', 'control_index': control_index, 'target_index': target_index})

    def CH(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Hadamard gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-H gate
        ch = H(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.append(ch)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CH', 'control_index': control_index, 'target_index': target_index})

    def CS(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Clifford-S gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-S gate
        cs = S(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CS gate to the circuit at the specified control and target qubits
        self.circuit.append(cs)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CS', 'control_index': control_index, 'target_index': target_index})

    def CT(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Clifford-T gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-T gate
        ct = T(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.append(ct)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CT', 'control_index': control_index, 'target_index': target_index})

    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a CRX gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the CRX gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Create a controlled-RX (CRX) gate with the specified angle
        crx_gate = Rx(rads=angle)(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CRX gate to the circuit
        self.circuit.append(crx_gate)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CRX', 'angle': angle, 'control_index': control_index, 'target_index': target_index})

    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a CRY gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the CRY gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Create a Controlled-RY gate with the specified angle
        cry = Ry(rads=angle)(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.append(cry)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CRY', 'angle': angle, 'control_index': control_index, 'target_index': target_index})

    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a CRZ gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the CRZ gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Create a Controlled-RZ gate with the specified angle
        crz = Rz(rads=angle)(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.append(crz)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CRZ', 'angle': angle, 'control_index': control_index, 'target_index': target_index})

    def CU3(self,
            angles: Iterable[float],
            control_index: int,
            target_index: int) -> None:
        """ Apply a CU3 gate to the circuit.

        Parameters
        ----------
        `angles` (Iterable[float]):
            The rotation angles in radians.
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-U3 gate with the specified angles
        cu3 = QasmUGate(angles[0] / np.pi, angles[1] / np.pi, angles[2] / np.pi)(self.qr[target_index]).controlled_by(self.qr[control_index])
        # Apply the CU3 gate to the circuit at the specified control and target qubits
        self.circuit.append(cu3)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CU3', 'angles': angles, 'control_index': control_index, 'target_index': target_index})

    def MCX(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a MCX gate to the circuit.

        Parameters
        ----------
        `control_indices (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCX gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(X(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCX', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCY(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a MCY gate to the circuit.

        Parameters
        ----------
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCY gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(Y(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCY', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCZ(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a MCZ gate to the circuit.

        Parameters
        ----------
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCZ gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(Z(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCZ', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCH(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-controlled Hadamard gate to the circuit.

        Parameters
        ----------
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCH gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(H(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCH', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCS(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-controlled Clifford-S gate to the circuit.

        Parameters
        ----------
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCS gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(S(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCS', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCT(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-controlled Clifford-T gate to the circuit.

        Parameters
        ----------
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCT gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(T(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCT', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCRX(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a MCRX gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the MCRX gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRX gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(Rx(rads=angle)(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCRX', 'angle': angle, 'control_indices': control_indices, 'target_indices': target_indices})

    def MCRY(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a MCRY gate to the circuit.

        Parameters
        ----------
        `angle (float):
            The rotation angle in radians.
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the MCRY gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRY gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(Ry(rads=angle)(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCRY', 'angle': angle, 'control_indices': control_indices, 'target_indices': target_indices})

    def MCRZ(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a MCRZ gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # If the angle is zero or a multiple of 2 pi, do not apply the MCRZ gate
        if angle == 0 or math.isclose(angle % (2 * np.pi), 0):
            return

        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(Rz(rads=angle)(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCRZ', 'angle': angle, 'control_indices': control_indices, 'target_indices': target_indices})

    def MCU3(self,
             angles: Iterable[float],
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a MCU3 gate to the circuit.

        Parameters
        ----------
        `angles` (Iterable[float]):
            The rotation angles in radians.
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to the length of control_indices with the specified angle
        cu3 = QasmUGate(angles[0] / np.pi, angles[1] / np.pi, angles[2] / np.pi)

        # Apply the MCU3 gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(cu3(self.qr[target_indices[i]]).controlled_by(*[self.qr[control_indices[j]] for j in range(len(control_indices))]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCU3', 'angles': angles, 'control_indices': control_indices, 'target_indices': target_indices})

    def measure(self,
                qubit_indices: int | Iterable[int]) -> None:
        """ Measure qubits in the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The indices of the qubits to measure.
        """
        # Measure the qubits
        self.circuit.append(cirq.measure(*[self.qr[qubit_indices[i]] for i in range(len(qubit_indices))], key='meas'))

        # Set the measurement as applied
        self.measured = True

        # Add the operation to the log
        self.circuit_log.append({'gate': 'measure', 'qubit_indices': qubit_indices})

    def get_statevector(self,
                        backend: Backend | None=None) -> Iterable[float]:
        """ Get the state vector of the circuit.

        Parameters
        ----------
        `backend` (Any | Backend):
            The backend to run the circuit on.

        Returns
        -------
        `statevector` (Iterable[float]): The state vector of the circuit.
        """
        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        self.change_lsb()

        if backend is None:
            # Define the state vector
            state_vector = self.circuit.final_state_vector(qubit_order=self.qr)

        else:
            # Run the circuit on the specified backend and define the state vector
            state_vector = backend.get_statevector(self.circuit)

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
                   backend: Backend | None=None) -> dict:
        """ Get the counts of the circuit.

        Parameters
        ----------
        `num_shots` (int):
            The number of shots to run.
        `backend` (Optional[object]):
            The backend to run the circuit on.

        Returns
        -------
        `counts` (dict): The counts of the circuit.
        """
        if backend is None:
            # Run the circuit
            state_vector = self.get_statevector()
            # Get the counts
            counts = {format(int(index), '0{}b'.format(self.num_qubits)): (abs(amplitude)**2 * num_shots)\
                      for index, amplitude in enumerate(state_vector)}

        else:
            # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
            self.change_lsb()
            # Run the circuit on the specified backend
            counts = backend.get_counts(self.circuit, num_shots)

        # Return the counts
        return counts

    def draw(self) -> None:
        """ Draw the circuit.
        """
        print(self.circuit)

    def get_depth(self) -> int:
        """ Get the depth of the circuit.

        Returns
        -------
        max_depth (int): The depth of the circuit.
        """
        # Since Cirq does not have a built-in depth method, we need to count the number of
        # operations applied to each qubit, and return the one with the highest count
        max_depth = 0

        # Count the number of operations applied to each qubit
        for qubit in range(self.num_qubits):
            depth = sum(1 for operation in self.circuit_log if
                        (isinstance(operation.get('qubit_indices', None), int) and qubit == operation['qubit_indices']) or
                        (isinstance(operation.get('control_indices', None), int) and qubit == operation['control_indices']) or
                        (isinstance(operation.get('target_indices', None), int) and qubit == operation['target_indices']) or
                        (isinstance(operation.get('qubit_indices', None), Iterable) and qubit in operation['qubit_indices']) or
                        (isinstance(operation.get('control_indices', None), Iterable) and qubit in operation['control_indices']) or
                        (isinstance(operation.get('target_indices', None), Iterable) and qubit in operation['target_indices']) or
                        qubit == operation.get('qubit_index', None) or
                        qubit == operation.get('control_index', None) or
                        qubit == operation.get('target_index', None))

            # Update the max depth
            if depth > max_depth:
                max_depth = depth

        # Return the max depth
        return max_depth

    def optimize(self) -> None:
        """ Transpile the circuit to CX and U3 gates.
        """
        # Convert the circuit to QiskitCircuit
        circuit = self.convert(QiskitCircuit)

        # Use the built-in transpiler from IBM Qiskit to transpile the circuit
        transpiled_circuit: qiskit.QuantumCircuit = transpile(circuit.circuit, basis_gates = ['cx', 'u3'])

        # Reset the circuit log (as we will be creating a new one given the transpiled circuit)
        self.circuit_log =[]

        # Iterate over the gates in the transpiled circuit
        for gate in transpiled_circuit.data:
            # Add the U3 gate to circuit log
            if gate[0].name == 'u3':
                self.circuit_log.append({'gate': 'U3', 'angles': gate[0].params, 'qubit_index': [gate[1][0]._index]})

            # Add the CX gate to circuit log
            else:
                self.circuit_log.append({'gate': 'CX', 'control_index': gate[1][0]._index, 'target_index': gate[1][1]._index})

        # Convert back to define the updated circuit after transpilation
        self.circuit = self.convert(type(self)).circuit

    def to_qasm(self) -> str:
        """ Convert the circuit to QASM.

        Returns
        -------
        `qasm` (str): The QASM representation of the circuit.
        """
        # Convert the circuit to QASM
        qasm = self.convert(QiskitCircuit).circuit.qasm()

        # Return the QASM
        return qasm