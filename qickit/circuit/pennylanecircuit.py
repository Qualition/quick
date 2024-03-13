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

__all__ = ['PennylaneCircuit']

from collections.abc import Iterable
import numpy as np
from numpy.typing import NDArray
import math

# Pennylane imports
import pennylane as qml

# Qiskit imports
import qiskit
from qiskit import transpile

# Import `qickit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.Backend`
from qickit.backend import Backend


class PennylaneCircuit(Circuit):
    """ `PennylaneCircuit` is the wrapper for using Xanadu's PennyLane in Qickit SDK.
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        """ Initialize a `qickit.PennylaneCircuit` instance.

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

        # Define the device
        self.device = qml.device("default.qubit", wires=self.num_qubits)

        # Define the circuit
        self.circuit = []

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
        rx = qml.RX
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.append(rx(angle, wires=qubit_index))

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
        ry = qml.RY
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.append(ry(angle, wires=qubit_index))

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
        rz = qml.RZ
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.append(rz(angle, wires=qubit_index))

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
        h = qml.Hadamard

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(h(wires=index))
        else:
            # If it's not an list, apply the H gate to the single qubit
            self.circuit.append(h(wires=qubit_indices))

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
        x = qml.PauliX

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(x(wires=index))
        else:
            # If it's not a list, apply the X gate to the single qubit
            self.circuit.append(x(wires=qubit_indices))

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
        y = qml.PauliY

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(y(wires=index))
        else:
            # If it's not a list, apply the Y gate to the single qubit
            self.circuit.append(y(wires=qubit_indices))

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
        z = qml.PauliZ

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(z(wires=index))
        else:
            # If it's not a list, apply the Z gate to the single qubit
            self.circuit.append(z(wires=qubit_indices))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'Z', 'qubit_indices': qubit_indices})

    def S(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Clifford-S gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Clifford S gate
        s = qml.S

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(s(wires=index))
        else:
            # If it's not a list, apply the S gate to the single qubit
            self.circuit.append(s(wires=qubit_indices))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'S', 'qubit_indices': qubit_indices})

    def T(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Clifford-T gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a Clifford T gate
        t = qml.T

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(t(wires=index))
        else:
            # If it's not a list, apply the T gate to the single qubit
            self.circuit.append(t(wires=qubit_indices))

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
        u3 = qml.U3
        # Apply the U3 gate to the circuit at the specified qubit
        self.circuit.append(u3(theta=angles[0], phi=angles[1], delta=angles[2], wires=qubit_index))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'U3', 'angles': angles, 'qubit_index': qubit_index})

    def CX(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Pauli-X gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-X gate
        cx = qml.CNOT
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.append(cx(wires=[control_index, target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CX', 'control_index': control_index, 'target_index': target_index})

    def CY(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Pauli-Y gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-Y gate
        cy = qml.CY
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.append(cy(wires=[control_index, target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CY', 'control_index': control_index, 'target_index': target_index})

    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Pauli-Z gate to the circuit.

        Parameters
        ----------
        `control_index` (int):
            The index of the control qubit.
        `target_index` (int):
            The index of the target qubit.
        """
        # Create a Controlled-Z gate
        cz = qml.CZ
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.append(cz(wires=[control_index, target_index]))

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
        ch = qml.CH
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.append(ch(wires=[control_index, target_index]))

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
        cs = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=control_index, wires=target_index)
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
        ct = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=control_index, wires=target_index)
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.append(ct)

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CT', 'control_index': control_index, 'target_index': target_index})

    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled RX gate to the circuit.

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

        # Create a Controlled-RX gate with the specified angle
        crx = qml.CRX
        # Apply the CRX gate to the circuit at the specified control and target qubits
        self.circuit.append(crx(angle, wires=[control_index, target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CRX', 'angle': angle, 'control_index': control_index, 'target_index': target_index})

    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled RY gate to the circuit.

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
        cry = qml.CRY
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.append(cry(angle, wires=[control_index, target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CRY', 'angle': angle, 'control_index': control_index, 'target_index': target_index})

    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled RZ gate to the circuit.

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
        crz = qml.CRZ
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.append(crz(angle, wires=[control_index, target_index]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CRZ', 'angle': angle, 'control_index': control_index, 'target_index': target_index})

    def CU3(self,
            angles: Iterable[float],
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled U3 gate to the circuit.

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
        cu3 = qml.U3(theta=angles[0], phi=angles[1], delta=angles[2],wires=0).matrix()
        # Apply the CU3 gate to the circuit at the specified control and target qubits
        self.circuit.append(qml.ControlledQubitUnitary(cu3, control_wires=control_index, wires=target_index))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'CU3', 'angles': angles, 'control_index': control_index, 'target_index': target_index})

    def MCX(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Pauli-X gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.PauliX(0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCX', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCY(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Pauli-Y gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.PauliY(0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCY', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCZ(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Pauli-Z gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.PauliZ(0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCZ', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCH(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Hadamard gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.Hadamard(0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCH', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCS(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Clifford-S gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCS', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCT(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Clifford-T gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCT', 'control_indices': control_indices, 'target_indices': target_indices})

    def MCRX(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled RX gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.RX(angle, wires=0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCRX', 'angle': angle, 'control_indices': control_indices, 'target_indices': target_indices})

    def MCRY(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled RY gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.RY(angle, wires=0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCRY', 'angle': angle, 'control_indices': control_indices, 'target_indices': target_indices})

    def MCRZ(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled RZ gate to the circuit.

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
            self.circuit.append(qml.ControlledQubitUnitary(qml.RZ(angle, wires=0).matrix(), control_wires=control_indices, wires=target_indices[i]))

        # Add the gate to the log
        self.circuit_log.append({'gate': 'MCRZ', 'angle': angle, 'control_indices': control_indices, 'target_indices': target_indices})

    def MCU3(self,
             angles: Iterable[float],
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled U3 gate to the circuit.

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

        # Apply the MCU3 gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            self.circuit.append(qml.ControlledQubitUnitary(qml.U3(theta=angles[0], phi=angles[1], delta=angles[2], wires=0).matrix(), control_wires=control_indices, wires=target_indices[i]))

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
        # In PennyLane, we apply measurements in '.run', '.get_statevector', and '.get_counts'
        # methods. This is due to the need for PennyLane quantum functions to return measurement results.
        # Therefore, we do not need to do anything here.
        pass

    def get_statevector(self,
                        backend: Backend | None=None) -> Iterable[float]:
        """ Get the state vector of the circuit.

        Parameters
        ----------
        `backend` (Backend | None=None):
            The backend to run the circuit on.

        Returns
        -------
        `statevector` (Iterable[float]): The state vector of the circuit.
        """
        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy()

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.change_lsb()

        def compile() -> qml.StateMP:
            """ Compile the circuit.

            Parameters
            ----------
            circuit (Iterable[qml.Op]):
                The list of operations representing the circuit.

            Returns
            -------
            (StateMP): The state vector of the circuit.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                qml.apply(op)

            # Return the measurement
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
        `backend` (Backend | None=None):
            The backend to run the circuit on.

        Returns
        -------
        `counts` (dict): The counts of the circuit.
        """
        # Set the seed
        np.random.seed(0)

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy()

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.change_lsb()

        def compile() -> Iterable[qml.ProbabilityMP]:
            """ Compile the circuit.

            Parameters
            ----------
            circuit (Iterable[qml.Op]):
                The list of operations representing the circuit.

            Returns
            -------
            (Iterable[qml.ProbabilityMP]): The list of probability measurements.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                qml.apply(op)

            # Return the measurement
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

        # Return the counts
        return counts

    def draw(self) -> None:
        """ Draw the circuit.
        """
        pass

    def get_depth(self) -> int:
        """ Get the depth of the circuit.

        Returns
        -------
        (int): The depth of the circuit.
        """
        @qml.qnode(self.device)
        def compile() -> Iterable[qml.ProbabilityMP]:
            """ Compiles the circuit.

            Parameters
            ----------
            circuit (Iterable[qml.Op]):
                The list of operations representing the circuit.

            Returns
            -------
            (Iterable[qml.ProbabilityMP]): The list of probability measurements.
            """
            # Apply the operations in the circuit
            for op in self.circuit:
                qml.apply(op)

            # Return the measurement
            return qml.expval(qml.PauliZ(0))

        # Get the depth of the circuit
        depth = qml.specs(compile)()['resources'].depth

        # Return the depth
        return depth

    def get_unitary(self) -> NDArray[np.number]:
        """ Get the unitary matrix of the circuit.

        Returns
        -------
        `unitary` (NDArray[np.number]): The unitary matrix of the circuit.
        """
        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy()

        def compile() -> None:
            """ Compile the circuit.

            Parameters
            ----------
            circuit (Iterable[qml.Op]):
                The list of operations representing the circuit.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                qml.apply(op)

        # Run the circuit and define the unitary matrix
        unitary = np.array(qml.matrix(compile)(), dtype=complex)

        # Return the unitary matrix
        return unitary

    def transpile(self) -> None:
        """ Transpile the circuit to U3 and CX gates.
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