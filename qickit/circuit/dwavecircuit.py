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

__all__ = ['DwaveCircuit']

from collections.abc import Iterable
import numpy as np
from numpy.typing import NDArray
import math
import matplotlib.pyplot as plt

# D-Wave imports
from dwave.gate import Circuit as DWCircuit
from dwave.gate.operations import *
from dwave.gate.simulator import simulate

# Qiskit imports
import qiskit
from qiskit import QuantumCircuit, transpile

# Import `qickit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit
from qickit.backend import Backend


class DwaveCircuit(Circuit):
    """ `DwaveCircuit` is the wrapper for using D-Wave's Gate-based framework
        in Qickit SDK.
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        """ Initialize a `qickit.DwaveCircuit` instance.

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

        # Define the circuit
        self.circuit = DWCircuit(self.num_qubits, self.num_clbits)

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
        rx = RX(angle)
        # Apply the RX gate to the circuit at the specified qubit
        with self.circuit.context as (q, c):
            rx(q[qubit_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        ry = RY(angle)
        # Apply the RY gate to the circuit at the specified qubit
        with self.circuit.context as (q, c):
            ry(q[qubit_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        rz = RZ(angle)
        # Apply the RZ gate to the circuit at the specified qubit
        with self.circuit.context as (q, c):
            rz(q[qubit_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        h = Hadamard()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                with self.circuit.context as (q, c):
                    h(q[index])
                # dwave.gate requires calling the unlock method to add more gates
                self.circuit.unlock()
        else:
            # If it's not an list, apply the H gate to the single qubit
            with self.circuit.context as (q, c):
                h(q[qubit_indices])
            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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
        x = X()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                with self.circuit.context as (q, c):
                    x(q[index])
                # dwave.gate requires calling the unlock method to add more gates
                self.circuit.unlock()
        else:
            # If it's not a list, apply the X gate to the single qubit
            with self.circuit.context as (q, c):
                x(q[qubit_indices])
            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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
        y = Y()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                with self.circuit.context as (q, c):
                    y(q[index])
                # dwave.gate requires calling the unlock method to add more gates
                self.circuit.unlock()
        else:
            # If it's not a list, apply the Y gate to the single qubit
            with self.circuit.context as (q, c):
                y(q[qubit_indices])
            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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
        z = Z()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                with self.circuit.context as (q, c):
                    z(q[index])
                # dwave.gate requires calling the unlock method to add more gates
                self.circuit.unlock()
        else:
            # If it's not a list, apply the Z gate to the single qubit
            with self.circuit.context as (q, c):
                z(q[qubit_indices])
            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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
        s = S()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                with self.circuit.context as (q, c):
                    s(q[index])
                # dwave.gate requires calling the unlock method to add more gates
                self.circuit.unlock()
        else:
            # If it's not a list, apply the S gate to the single qubit
            with self.circuit.context as (q, c):
                s(q[qubit_indices])
            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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
        t = T()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Iterable):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                with self.circuit.context as (q, c):
                    t(q[index])
                # dwave.gate requires calling the unlock method to add more gates
                self.circuit.unlock()
        else:
            # If it's not a list, apply the T gate to the single qubit
            with self.circuit.context as (q, c):
                t(q[qubit_indices])
            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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
        u3 = Rotation(angles)
        # Apply the U3 gate to the circuit at the specified qubit
        with self.circuit.context as (q, c):
                u3(q[qubit_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        cx = CX()
        # Apply the CX gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                cx(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        cy = CY()
        # Apply the CY gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                cy(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        cz = CZ()
        # Apply the CZ gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                cz(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        # Create a Controlled Hadamard gate with the specified angle
        ch = CHadamard()
        # Apply the CRX gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                ch(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        # As of now, dwave.gate does not support CS gates. We can use the ControlledOperation class
        # to create a custom CS gate
        class CSControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = S()

        # Apply the CS gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
            CSControlledOp(control=q[control_index], target=q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        # As of now, dwave.gate does not support CT gates. We can use the ControlledOperation class
        # to create a custom CT gate
        class CTControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = T()

        # Apply the CT gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
            CTControlledOp(control=q[control_index], target=q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        crx = CRX(angle)
        # Apply the CRX gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                crx(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        cry = CRY(angle)
        # Apply the CRY gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                cry(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        crz = CRZ(angle)
        # Apply the CRz gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
                crz(q[control_index], q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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
        # As of now, dwave.gate does not support CU3 gates. We can use the ControlledOperation class
        # to create a custom CU3 gate
        class CU3ControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = Rotation(angles)

        # Apply the CU3 gate to the circuit at the specified control and target qubits
        with self.circuit.context as (q, c):
            CU3ControlledOp(control=q[control_index], target=q[target_index])

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCY gates. We can use the ControlledOperation class
        # to create a custom MCY gate
        class MCXControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = X()

        # Apply the MCY gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCXControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCY gates. We can use the ControlledOperation class
        # to create a custom MCY gate
        class MCYControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = Y()

        # Apply the MCY gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCYControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCZ gates. We can use the ControlledOperation class
        # to create a custom MCZ gate
        class MCZControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = Z()

        # Apply the MCZ gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCZControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCH gates. We can use the ControlledOperation class
        # to create a custom MCH gate
        class MCHControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = Hadamard()

        # Apply the MCH gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCHControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCS gates. We can use the ControlledOperation class
        # to create a custom MCS gate
        class MCSControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = S()

        # Apply the MCS gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCSControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCT gates. We can use the ControlledOperation class
        # to create a custom MCT gate
        class MCTControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = Hadamard()

        # Apply the MCT gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCTControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCRX gates. We can use the ControlledOperation class
        # to create a custom MCRX gate
        class MCRXControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = RX(angle)

        # Apply the MCRX gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCRXControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCRY gates. We can use the ControlledOperation class
        # to create a custom MCRY gate
        class MCRYControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = RY(angle)

        # Apply the MCRY gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCRYControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCRZ gates. We can use the ControlledOperation class
        # to create a custom MCRZ gate
        class MCRZControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = RZ(angle)

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCRZControlledOp(control=([q[index] for index in control_indices]), target=q[i])

            # dwave.gate requires calling the unlock method to add more gates
            self.circuit.unlock()

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

        # As of now, dwave.gate does not support MCU3 gates. We can use the ControlledOperation class
        # to create a custom MCU3 gate
        class MCU3ControlledOp(ControlledOperation):
            _num_control = len(control_indices)
            _num_target = 1
            _target_operation = Rotation(angles)

        # Apply the MCU3 gate to the circuit at the control and target qubits
        for i in range(len(target_indices)):
            with self.circuit.context as (q, c):
                MCU3ControlledOp(control=([q[index] for index in control_indices]), target=q[i])

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
        with self.circuit.context as (q, c):
            meas = Measurement(q) | c

        # dwave.gate requires calling the unlock method to add more gates
        self.circuit.unlock()

        # Add the measurement to the class
        self.measurement = meas

        # Add the measurement to the log
        self.circuit_log.append({'gate': 'measure', 'qubit_indices': qubit_indices})

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
        circuit: DwaveCircuit = self.copy()

        # D-wave uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.change_lsb()

        if backend is None:
            # Run the circuit
            simulate(circuit.circuit)
            # Define the state vector
            state_vector = np.real(circuit.circuit.state)

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
        if self.measured is False:
            self.measure(range(self.num_qubits))

        # Copy the circuit as the operations are applied inplace
        circuit: DwaveCircuit = self.copy()

        # D-wave uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.change_lsb()

        if backend is None:
            # Run the circuit
            simulate(circuit.circuit)
            # Execute the circuit on the backend
            counts = self.measurement.sample(range(self.num_qubits), num_shots)

        else:
            # Execute the circuit on the specified backend
            counts = backend.get_counts(circuit, num_shots)

        # Return the counts
        return counts

    def draw(self) -> plt.figure:
        """ Draw the circuit.
        """
        # `dwave.gate` does not support drawing circuits. We can use the `qiskit` library to draw the circuit.
        # Convert the circuit to qasm format
        qasm = self.circuit.to_qasm()

        # Draw the circuit
        qc = QuantumCircuit.from_qasm_str(qasm)

        # Return the circuit
        return qc.draw(output='mpl')

    def get_depth(self) -> int:
        """ Get the depth of the circuit.

        Returns
        -------
        (int): The depth of the circuit.
        """
        return len(self.circuit.circuit)

    def get_unitary(self) -> NDArray[np.number]:
        """ Get the unitary matrix of the circuit.

        Returns
        -------
        `unitary` (NDArray[np.number]): The unitary matrix of the circuit.
        """
        # Copy the circuit as the operations are applied inplace
        circuit = self.convert(QiskitCircuit)

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

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