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

__all__ = ['Circuit']

from abc import ABC, abstractmethod
from collections.abc import Iterable
from types import NotImplementedType
import numpy as np
from numpy.typing import NDArray

# Qiskit imports
import qiskit
from qiskit import transpile

# `qickit.Backend` import
from qickit.backend import Backend


class Circuit(ABC):
    """ `qickit.Circuit` is the class for creating and manipulating gate-based circuits.
        This class is defined for external Quantum Circuit (QC) Frameworks. Current
        supported packages are :
        - IBM Qiskit
        - Google's Cirq
        - Quantinuum's PyTKET
        - Xanadu's PennyLane
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        """ Initialize a `qickit.Circuit` instance.

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
        self.circuit = None
        # Define the measurement status
        self.measured = False
        # Define the circuit log (list[dict])
        self.circuit_log = []

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def H(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Hadamard gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        pass

    @abstractmethod
    def X(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Pauli-X gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        pass

    @abstractmethod
    def Y(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Pauli-Y gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        pass

    @abstractmethod
    def Z(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Pauli-Z gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        pass

    @abstractmethod
    def S(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Clifford-S gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        pass

    @abstractmethod
    def T(self,
          qubit_indices: int | Iterable[int]) -> None:
        """ Apply a Clifford-T gate to the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def MCX(self,
            control_indices: int | Iterable[int],
            target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled Pauli-X gate to the circuit.

        Parameters
        ----------
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def MCRY(self,
             angle: float,
             control_indices: int | Iterable[int],
             target_indices: int | Iterable[int]) -> None:
        """ Apply a Multi-Controlled RY gate to the circuit.

        Parameters
        ----------
        `angle` (float):
            The rotation angle in radians.
        `control_indices` (int | Iterable[int]):
            The index of the control qubit(s).
        `target_indices` (int | Iterable[int]):
            The index of the target qubit(s).
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def unitary(self,
                unitary_matrix: NDArray[np.number],
                qubit_indices:  int | Iterable[int]) -> None:
        """ Apply a unitary gate to the circuit.

        Parameters
        ----------
        `unitary_matrix` (NDArray[np.number]):
            The unitary matrix to apply to the circuit.
        `qubit_indices` (int | Iterable[int]):
            The index of the qubit(s) to apply the gate to.
        """
        # Create a qiskit circuit
        circuit = qiskit.QuantumCircuit(self.num_qubits, self.num_clbits)

        # Apply the unitary matrix to the circuit
        circuit.unitary(unitary_matrix, qubit_indices)

        # Transpile the unitary operator to a series of CX and U3 gates
        transpiled_circuit = transpile(circuit, basis_gates = ['cx', 'u3'])

        # Iterate over the gates in the transpiled circuit
        for gate in transpiled_circuit.data:
            # Add the U3 gate
            if gate[0].name == 'u3':
                self.U3(gate[0].params, gate[1][0]._index)

            # Add the CX gate
            else:
                self.CX(gate[1][0].index, gate[1][1]._index)

    def vertical_reverse(self) -> None:
        """ Perform a vertical reverse operation.
        """
        # Iterate over every operation, and change the index accordingly
        for operation in self.circuit_log:
            keys = ['target_indices', 'control_indices', 'qubit_indices', 'qubit_index', 'control_index', 'target_index']
            for key in keys:
                if key in operation:
                    if isinstance(operation[key], Iterable):
                        operation[key] = [(self.num_qubits - 1 - index) for index in operation[key]]
                    else:
                        operation[key] = (self.num_qubits - 1 - operation[key])

        # Update the circuit
        self.circuit = self.convert(type(self)).circuit

    def horizontal_reverse(self,
                           adjoint: bool=True) -> None:
        """ Perform a horizontal reverse operation.

        Parameters
        ----------
        `adjoint` (bool):
            Whether or not to apply the adjoint of the circuit.
        """
        # Reverse the order of the operations
        self.circuit_log = self.circuit_log[::-1]

        # If adjoint is True, then multiply the angles by -1
        if adjoint:
            for operation in self.circuit_log:
                if 'angle' in operation:
                    operation['angle'] = -operation['angle']
                elif 'angles' in operation:
                    operation['angles'] = [-angle for angle in operation['angles']]

        # Update the circuit
        self.circuit = self.convert(type(self)).circuit

    def add(self,
            circuit: Circuit,
            qubit_indices: int | Iterable[int]) -> None:
        """ Append two circuits together in a sequence.

        Parameters
        ----------
        `circuit` (Circuit):
            The circuit to append to the current circuit.
        `qubit_indices` (int | Iterable[int]):
            The indices of the qubits to add the circuit to.
        """
        # The number of qubits must match the number of qubits in the circuit.
        assert len(qubit_indices) == circuit.num_qubits, "The number of qubits must match the number of qubits in the circuit."

        # Update the qubit indices
        for operation in circuit.circuit_log:
            keys = ['target_indices', 'control_indices', 'qubit_indices', 'qubit_index', 'control_index', 'target_index']
            for key in keys:
                if key in operation:
                    if isinstance(operation[key], Iterable):
                        operation[key] = [(qubit_indices[index]) for index in operation[key]]
                    else:
                        operation[key] = (qubit_indices[operation[key]])

        # Add the other circuit's log to the circuit log
        self.circuit_log.extend(circuit.circuit_log)

        # Create the updated circuit
        self.circuit = self.convert(type(self)).circuit

    @abstractmethod
    def measure(self,
                qubit_indices: int | Iterable[int]) -> None:
        """ Measure the qubits in the circuit.

        Parameters
        ----------
        `qubit_indices` (int | Iterable[int]):
            The indices of the qubits to measure.
        """
        pass

    @abstractmethod
    def get_statevector(self,
                        backend: Backend | None=None) -> Iterable[float]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `backend` (Any | Backend):
            The backend to run the circuit on.

        Returns
        -------
        `statevector` (Iterable[float]): The statevector of the circuit.
        """
        pass

    @abstractmethod
    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None=None) -> dict:
        """ Get the counts of the circuit.

        Parameters
        ----------
        `num_shots` (int):
            The number of shots to run.
        `backend` (Any | Backend):
            The backend to run the circuit on.

        Returns
        -------
        `counts` (dict): The counts of the circuit.
        """
        pass

    @abstractmethod
    def draw(self) -> None:
        """ Draw the circuit.
        """
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """ Get the depth of the circuit.

        Returns
        -------
        `depth` (int): The depth of the circuit.
        """
        pass

    @abstractmethod
    def get_unitary(self) -> NDArray[np.number]:
        """ Get the unitary matrix of the circuit.

        Returns
        -------
        `unitary` (NDArray[np.number]): The unitary matrix of the circuit.
        """
        pass

    @abstractmethod
    def transpile(self) -> None:
        """ Transpile the circuit to U3 and CX gates.
        """
        pass

    def compress(self,
                 compression_percentage: float) -> None:
        """ Compresses a `qickit.Circuit` object.

        Parameters
        ----------
        `compression_percentage` (float):
            The percentage of compression. Value between 0.0 to 1.0.
        """
        # Define angle closeness threshold
        threshold = np.pi * compression_percentage

        # Initialize a list for the indices that will be removed
        indices_to_remove = []

        # Iterate over all angles, and set the angles within the
        # compression percentage to 0
        for index, operation in enumerate(self.circuit_log):
                if 'angle' in operation:
                    if abs(operation['angle']) < threshold:
                        indices_to_remove.append(index)

                elif 'angles' in operation:
                    if all([abs(angle) < threshold for angle in operation['angles']]):
                        indices_to_remove.append(index)

        # Remove the operations with angles within the compression percentage
        for index in sorted(indices_to_remove, reverse=True):
            del self.circuit_log[index]

        # Update the circuit
        self.circuit = self.convert(type(self)).circuit

    def change_mapping(self,
                       qubit_indices: Iterable[int]) -> None:
        """ Change the mapping of the circuit.

        Parameters
        ----------
        `qubit_indices` (Iterable[int]):
            The updated order of the qubits.
        """
        # The number of qubits must match the number of qubits in the circuit.
        assert len(qubit_indices) == self.num_qubits, "The number of qubits must match the number of qubits in the circuit."

        # Update the qubit indices
        for operation in self.circuit_log:
            keys = ['target_indices', 'control_indices', 'qubit_indices', 'qubit_index', 'control_index', 'target_index']
            for key in keys:
                if key in operation:
                    if isinstance(operation[key], Iterable):
                        operation[key] = [(qubit_indices[index]) for index in operation[key]]
                    else:
                        operation[key] = (qubit_indices[operation[key]])

        # Convert the circuit to create the updated circuit
        new_circuit = self.convert(Circuit)

        # Update the circuit
        self.circuit = new_circuit.circuit

    def convert(self,
                circuit_framework: Circuit) -> Circuit:
        """ Convert the circuit to another circuit framework.

        Parameters
        ----------
        `circuit_framework` (Circuit):
            The circuit framework to convert to.

        Returns
        -------
        `converted_circuit` (Circuit): The converted circuit.
        """
        # Define the new circuit using the provided framework
        converted_circuit = circuit_framework(self.num_qubits, self.num_clbits)

        # Define a mapping between Qiskit gate names and corresponding methods in the target framework
        gate_mapping = {
            'RX': converted_circuit.RX,
            'RY': converted_circuit.RY,
            'RZ': converted_circuit.RZ,
            'H': converted_circuit.H,
            'X': converted_circuit.X,
            'Y': converted_circuit.Y,
            'Z': converted_circuit.Z,
            'S': converted_circuit.S,
            'T': converted_circuit.T,
            'U3': converted_circuit.U3,
            'CX': converted_circuit.CX,
            'CY': converted_circuit.CY,
            'CZ': converted_circuit.CZ,
            'CH': converted_circuit.CH,
            'CS': converted_circuit.CS,
            'CT': converted_circuit.CT,
            'CRX': converted_circuit.CRX,
            'CRY': converted_circuit.CRY,
            'CRZ': converted_circuit.CRZ,
            'CU3': converted_circuit.CU3,
            'MCX': converted_circuit.MCX,
            'MCY': converted_circuit.MCY,
            'MCZ': converted_circuit.MCZ,
            'MCH': converted_circuit.MCH,
            'MCS': converted_circuit.MCS,
            'MCT': converted_circuit.MCT,
            'MCRX': converted_circuit.MCRX,
            'MCRY': converted_circuit.MCRY,
            'MCRZ': converted_circuit.MCRZ,
            'MCU3': converted_circuit.MCU3,
            'measure': converted_circuit.measure
        }

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in self.circuit_log:
            # Find gate name
            gate_name = gate_info['gate']

            # Slide dict to keep kwargs only
            gate_info = dict(list(gate_info.items())[1:])

            # Use the gate mapping to apply the corresponding gate
            gate_mapping[gate_name](**gate_info)

        # Return the converted circuit
        return converted_circuit

    @abstractmethod
    def to_qasm(self) -> str:
        """ Convert the circuit to QASM.

        Returns
        -------
        `qasm` (str): The QASM representation of the circuit.
        """
        pass

    def reset(self) -> None:
        """ Reset the circuit to an empty circuit.
        """
        # Reset the circuit log
        self.circuit_log = []

        # Add a zero angle RX gate (so to not raise an empty list error)
        self.RX(0, 0)

        # Convert back to update the circuit
        self.circuit = self.convert(type(self)).circuit

    def __eq__(self,
               other_circuit: Circuit) -> bool:
        """ Compare two circuits for equality.

        Parameters
        ----------
        `other_circuit` (Circuit):
            The other circuit to compare to.

        Returns
        -------
        (bool): Whether the two circuits are equal.
        """
        return self.circuit_log == other_circuit.circuit_log

    def __len__(self) -> int:
        """ Get the number of the circuit operations.

        Returns
        -------
        (int): The number of the circuit operations.
        """
        return len(self.circuit_log)

    def __str__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        (str): The string representation of the circuit.
        """
        return str(self.circuit_log)

    def __repr__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        (str): The string representation of the circuit.
        """
        return f"Circuit(num_qubits={self.num_qubits}, num_clbits={self.num_clbits})"

    @classmethod
    def __subclasscheck__(cls, C) -> bool:
        """ Checks if a class is a `qickit.Circuit` if the class
        passed does not directly inherit from `qickit.Circuit`.

        Parameters
        ----------
        `C` (type):
            The class to check if it is a subclass.

        Returns
        -------
        (bool): Whether or not the class is a subclass.
        """
        if cls is Circuit:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return False

    @classmethod
    def __subclasshook__(cls, C) -> bool | NotImplementedType:
        """ Checks if a class is a `qickit.Circuit` if the class
        passed does not directly inherit from `qickit.Circuit`.

        Parameters
        ----------
        `C` (type):
            The class to check if it is a subclass.

        Returns
        -------
        (bool | NotImplementedType): Whether or not the class is a subclass.
        """
        if cls is Circuit:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return NotImplemented

    @classmethod
    def __instancecheck__(cls, C) -> bool:
        """ Checks if an object is a `qickit.Circuit` given its
        interface.

        Parameters
        ----------
        `C` (object):
            The instance to check.

        Returns
        -------
        (bool): Whether or not the instance is a `qickit.Circuit`.
        """
        if cls is Circuit:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return False