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

import pytket.circuit

__all__ = ['Circuit']

from abc import ABC, abstractmethod
import copy
from functools import wraps
import inspect
from itertools import islice
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from numpy.typing import NDArray
import re
from types import NotImplementedType
from typing import Callable, Type, TYPE_CHECKING

# Qiskit imports
import qiskit # type: ignore

# Cirq imports
import cirq # type: ignore

# Pennylane imports
import pennylane as qml # type: ignore

# PyTKET imports
import pytket

# Import `qickit.backend.Backend`
# import `qickit.synthesis.unitarypreparation.QiskitTranspiler`
if TYPE_CHECKING:
    from qickit.backend import Backend

from qickit.synthesis.unitarypreparation import QiskitTranspiler

# Import `qickit.types.collection.Collection` and `qickit.types.circuit_type.Circuit_Type`
from qickit.types import Collection, Circuit_Type

""" Set the frozensets for the keys to be used:
- Decorator `Circuit.gatemethod()`
- Method `Circuit.vertical_reverse()`
- Method `Circuit.horizontal_reverse()`
- Method `Circuit.add()`
- Method `Circuit.change_mapping()`
"""
QUBIT_KEYS = frozenset(['qubit_index', 'control_index', 'target_index', 'first_qubit',
                        'second_qubit', 'first_target_index', 'second_target_index'])
QUBIT_LIST_KEYS = frozenset(['qubit_indices', 'control_indices', 'target_indices'])
ANGLE_KEYS = frozenset(['angle', 'angles'])


class Circuit(ABC):
    """ `qickit.circuit.Circuit` is the class for creating and manipulating gate-based circuits.
    This class is defined for external Quantum Circuit (QC) Frameworks.
    Current supported packages are :
    - IBM Qiskit
    - Google's Cirq
    - Quantinuum's PyTKET
    - Xanadu's PennyLane

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
    `circuit` : Circuit_Type
        The circuit framework type.
    `measured` : bool
        The measurement status of the circuit.
    `circuit_log` : list[dict]
        The log of the circuit operations.

    Raises
    ------
    TypeError
        Number of qubits and classical bits must be integers.
    ValueError
        Number of qubits and classical bits must be greater than 0.

    Usage
    -----
    >>> circuit = Circuit(num_qubits=2, num_clbits=2)
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        """ Initialize a `qickit.circuit.Circuit` instance.
        """
        if not isinstance(num_qubits, int) or not isinstance(num_clbits, int):
            raise TypeError("Number of qubits and classical bits must be integers.")

        if num_qubits < 1 or num_clbits < 1:
            raise ValueError("Number of qubits and classical bits must be greater than 0.")

        # Define the number of quantum bits
        self.num_qubits = num_qubits
        # Define the number of classical bits
        self.num_clbits = num_clbits
        # Define the circuit
        self.circuit = Circuit_Type
        # Define the measurement status
        self.measured = False
        # Define the circuit log (list[dict])
        self.circuit_log: list[dict] = []

    @staticmethod
    def gatemethod(method):
        """ Decorator for gate methods. This decorator logs the operations,
        and catches any errors in the gate parameters.

        Parameters
        ----------
        method : Callable[P, None]
            The method to decorate.

        Returns
        -------
        Callable[P, None]
            The decorated method.

        Raises
        ------
        TypeError
            Qubit index must be an integer.
            Angle must be a float or integer.
        ValueError
            Qubit index out of range.

        Usage
        -----
        >>> @Circuit.gatemethod
        ... def RX(self, angle: float, qubit_index: int) -> None:
        ...     ...
        """
        @wraps(method)
        def wrapped(instance,
                    *args,
                    **kwargs) -> None:
            # Retrieve the method's signature, including 'self' for instance methods
            sig = inspect.signature(method)
            # Bind the provided arguments to the method's parameters
            bound_args = sig.bind(instance, *args, **kwargs)
            bound_args.apply_defaults()

            # Prepare a dictionary to log the method name and its arguments
            params = {}

            # Populate the log dictionary with the method's arguments
            for name, value in islice(bound_args.arguments.items(), 1, None):
                # Convert range objects to lists
                if isinstance(value, range):
                    value = list(value)

                # Convert `np.integer` instances to int
                elif isinstance(value, np.integer):
                    value = int(value)

                # Convert `np.float` instances to float
                elif isinstance(value, np.floating):
                    value = float(value)

                # Ensure indices are valid indices (less than number of qubits - 1)
                # If index is negative, it is counted from the end of the list
                if name in QUBIT_KEYS:
                    if isinstance(value, Collection):
                        value = value[0]
                    if not isinstance(value, int):
                        raise TypeError("Qubit index must be an integer.")
                    if value >= instance.num_qubits:
                        raise ValueError("Qubit index out of range.")
                    if value < 0:
                        value = instance.num_qubits + value

                if name in QUBIT_LIST_KEYS:
                    if isinstance(value, Collection):
                        for i, index in enumerate(value):
                            if not isinstance(index, int):
                                raise TypeError("Qubit index must be an integer.")
                            if index >= instance.num_qubits:
                                raise ValueError("Qubit index out of range.")
                            if index < 0:
                                value[i] = instance.num_qubits + index
                    if isinstance(value, int):
                        if value >= instance.num_qubits:
                            raise ValueError("Qubit index out of range.")
                        if value < 0:
                            value = instance.num_qubits + value

                # Ensure that angles are valid
                # Don't apply any gates if the angle is 0 or an integer multiple of 2*pi
                if name in ANGLE_KEYS:
                    if isinstance(value, Collection):
                        for angle in value:
                            if not isinstance(angle, (int, float)):
                                raise TypeError("Angle must be a number.")
                    else:
                        if not isinstance(value, (int, float)):
                            raise TypeError("Angle must be a number.")
                        if value == 0 or np.isclose(value % (2 * np.pi), 0):
                            return

                params[name] = value

            # Append the method log to the instance's circuit log
            instance.circuit_log.append({'gate': method.__name__} | params)

            return method(instance, **params)

        return wrapped

    @abstractmethod
    def Identity(self,
                 qubit_indices: int | Collection[int]) -> None:
        """ Apply an Identity gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.Identity(qubit_indices=0)
        >>> circuit.Identity(qubit_indices=[0, 1])
        """

    @abstractmethod
    def X(self,
          qubit_indices: int | Collection[int]) -> None:
        """ Apply a Pauli-X gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.X(qubit_indices=0)
        >>> circuit.X(qubit_indices=[0, 1])
        """

    @abstractmethod
    def Y(self,
          qubit_indices: int | Collection[int]) -> None:
        """ Apply a Pauli-Y gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.Y(qubit_indices=0)
        >>> circuit.Y(qubit_indices=[0, 1])
        """

    @abstractmethod
    def Z(self,
          qubit_indices: int | Collection[int]) -> None:
        """ Apply a Pauli-Z gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.Z(qubit_indices=0)
        >>> circuit.Z(qubit_indices=[0, 1])
        """

    @abstractmethod
    def H(self,
          qubit_indices: int | Collection[int]) -> None:
        """ Apply a Hadamard gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.H(qubit_indices=0)
        >>> circuit.H(qubit_indices=[0, 1])
        """

    @abstractmethod
    def S(self,
          qubit_indices: int | Collection[int]) -> None:
        """ Apply a Clifford-S gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.S(qubit_indices=0)
        >>> circuit.S(qubit_indices=[0, 1])
        """

    @abstractmethod
    def T(self,
          qubit_indices: int | Collection[int]) -> None:
        """ Apply a Clifford-T gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Usage
        -----
        >>> circuit.T(qubit_indices=0)
        >>> circuit.T(qubit_indices=[0, 1])
        """

    @abstractmethod
    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        """ Apply a RX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_index` : int
            The index of the qubit to apply the gate to.

        Usage
        -----
        >>> circuit.RX(angle=np.pi/2, qubit_index=0)
        """

    @abstractmethod
    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        """ Apply a RY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_index` : int
            The index of the qubit to apply the gate to.

        Usage
        -----
        >>> circuit.RY(angle=np.pi/2, qubit_index=0)
        """

    @abstractmethod
    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        """ Apply a RZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_index` : int
            The index of the qubit to apply the gate to.

        Usage
        -----
        >>> circuit.RZ(angle=np.pi/2, qubit_index=0)
        """

    @abstractmethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        """ Apply a U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Collection[float]
            The rotation angles in radians.
        `qubit_index` : int
            The index of the qubit to apply the gate to.

        Usage
        -----
        >>> circuit.U3(angles=[np.pi/2, np.pi/2, np.pi/2], qubit_index=0)
        """

    @abstractmethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        """ Apply a SWAP gate to the circuit.

        Parameters
        ----------
        `first_qubit` : int
            The index of the first qubit.
        `second_qubit` : int
            The index of the second qubit.

        Usage
        -----
        >>> circuit.SWAP(first_qubit=0, second_qubit=1)
        """

    @abstractmethod
    def CX(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Pauli-X gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CX(control_index=0, target_index=1)
        """

    @abstractmethod
    def CY(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Pauli-Y gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CY(control_index=0, target_index=1)
        """

    @abstractmethod
    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Pauli-Z gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CZ(control_index=0, target_index=1)
        """

    @abstractmethod
    def CH(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Hadamard gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CH(control_index=0, target_index=1)
        """

    @abstractmethod
    def CS(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Clifford-S gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CS(control_index=0, target_index=1)
        """

    @abstractmethod
    def CT(self,
           control_index: int,
           target_index: int) -> None:
        """ Apply a Controlled Clifford-T gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CT(control_index=0, target_index=1)
        """

    @abstractmethod
    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled RX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CRX(angle=np.pi/2, control_index=0, target_index=1)
        """

    @abstractmethod
    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled RY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CRY(angle=np.pi/2, control_index=0, target_index=1)
        """

    @abstractmethod
    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled RZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CRZ(angle=np.pi/2, control_index=0, target_index=1)
        """

    @abstractmethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        """ Apply a Controlled U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Collection[float]
            The rotation angles in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Usage
        -----
        >>> circuit.CU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_index=0, target_index=1)
        """

    @abstractmethod
    def CSWAP(self,
              control_index: int,
              first_target_index: int,
              second_target_index: int) -> None:
        """ Apply a Controlled SWAP gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Usage
        -----
        >>> circuit.CSWAP(control_index=0, first_target_index=1, second_target_index=2)
        """

    @abstractmethod
    def MCX(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled Pauli-X gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCX(control_indices=0, target_indices=1)
        >>> circuit.MCX(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCX(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCX(control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCY(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled Pauli-Y gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCY(control_indices=0, target_indices=1)
        >>> circuit.MCY(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCY(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCY(control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCZ(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled Pauli-Z gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCZ(control_indices=0, target_indices=1)
        >>> circuit.MCZ(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCZ(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCZ(control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCH(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled Hadamard gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCH(control_indices=0, target_indices=1)
        >>> circuit.MCH(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCH(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCH(control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCS(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled Clifford-S gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCS(control_indices=0, target_indices=1)
        >>> circuit.MCS(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCS(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCS(control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCT(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled Clifford-T gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCT(control_indices=0, target_indices=1)
        >>> circuit.MCT(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCT(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCT(control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCRX(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled RX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCRX(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCRX(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCRX(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCRX(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCRY(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled RY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCRY(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCRY(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCRY(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCRY(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCRZ(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled RZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        """ Apply a Multi-Controlled U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Collection[float]
            The rotation angles in radians.
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).

        Usage
        -----
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=0, target_indices=1)
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=0, target_indices=[1, 2])
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=[0, 1], target_indices=2)
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=[0, 1], target_indices=[2, 3])
        """

    @abstractmethod
    def MCSWAP(self,
               control_indices: int | Collection[int],
               first_target_index: int,
               second_target_index: int) -> None:
        """ Apply a Controlled SWAP gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Usage
        -----
        >>> circuit.MCSWAP(control_indices=0, first_target_index=1, second_target_index=2)
        >>> circuit.MCSWAP(control_indices=[0, 1], first_target_index=2, second_target_index=3)
        """

    @abstractmethod
    def GlobalPhase(self,
                    angle: float) -> None:
        """ Apply a global phase to the circuit.

        Parameters
        ----------
        `angle` : float
            The global phase to apply to the circuit.

        Usage
        -----
        >>> circuit.GlobalPhase(angle=np.pi/2)
        """

    def unitary(self,
                unitary_matrix: NDArray[np.number],
                qubit_indices:  int | Collection[int]) -> None:
        """ Apply a unitary gate to the circuit.

        Parameters
        ----------
        `unitary_matrix` : NDArray[np.number]
            The unitary matrix to apply to the circuit.
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        ValueError
            The unitary matrix must have a size of 2^n x 2^n, where n is the number of qubits.
            The unitary matrix must be unitary.
            The number of qubits passed must be the same as the number of qubits needed to prepare the unitary.

        Usage
        -----
        >>> circuit.unitary([[0, 1], [1, 0]], qubit_indices=0)
        >>> circuit.unitary([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], qubit_indices=[0, 1])
        """
        # Initialize the unitary preparation schema
        unitary_preparer = QiskitTranspiler(type(self))

        # Prepare the unitary matrix
        circuit = unitary_preparer.prepare_unitary(unitary_matrix, qubit_indices)

        # Add the circuit to the current circuit
        self.add(circuit, qubit_indices)

    def vertical_reverse(self) -> None:
        """ Perform a vertical reverse operation.

        Usage
        -----
        >>> circuit.vertical_reverse()
        """
        # Iterate over every operation, and change the index accordingly
        for operation in self.circuit_log:
            keys = QUBIT_KEYS.union(QUBIT_LIST_KEYS)
            for key in keys:
                if key in operation:
                    if isinstance(operation[key], Collection):
                        operation[key] = [(self.num_qubits - 1 - index) for index in operation[key]]
                    else:
                        operation[key] = (self.num_qubits - 1 - operation[key])

        # Update the circuit
        self.circuit = self.convert(type(self)).circuit

    def horizontal_reverse(self,
                           adjoint: bool = True) -> None:
        """ Perform a horizontal reverse operation.

        Parameters
        ----------
        `adjoint` : bool
            Whether or not to apply the adjoint of the circuit.

        Raises
        ------
        TypeError
            Adjoint must be a boolean.

        Usage
        -----
        >>> circuit.horizontal_reverse()
        >>> circuit.horizontal_reverse(adjoint=True)
        """
        # Check if the adjoint is a boolean
        if not isinstance(adjoint, bool):
            raise TypeError("Adjoint must be a boolean.")

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
            qubit_indices: int | Collection[int]) -> None:
        """ Append two circuits together in a sequence.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to append to the current circuit.
        `qubit_indices` : int | Collection[int]
            The indices of the qubits to add the circuit to.

        Raises
        ------
        TypeError
            The circuit must be a Circuit object.
        ValueError
            The number of qubits must match the number of qubits in the circuit.

        Usage
        -----
        >>> circuit.add(circuit=circuit2, qubit_indices=0)
        >>> circuit.add(circuit=circuit2, qubit_indices=[0, 1])
        """
        # Check if the circuit is a Circuit object
        if not isinstance(circuit, Circuit):
            raise TypeError("The circuit must be a Circuit object.")

        # Convert the qubit indices to a list if it is a range
        if isinstance(qubit_indices, range):
            qubit_indices = list(qubit_indices)

        if isinstance(qubit_indices, Collection):
            # The number of qubits must match the number of qubits in the circuit.
            if len(qubit_indices) != circuit.num_qubits:
                raise ValueError("The number of qubits must match the number of qubits in the circuit.")

        # Create a copy of the circuit
        circuit = copy.deepcopy(circuit)

        # Update the qubit indices
        for operation in circuit.circuit_log:
            keys = QUBIT_KEYS.union(QUBIT_LIST_KEYS)
            for key in keys:
                if key in operation:
                    if isinstance(operation[key], Collection):
                        if isinstance(qubit_indices, Collection):
                            operation[key] = [(qubit_indices[index]) for index in operation[key]]
                    else:
                        if isinstance(qubit_indices, Collection):
                            operation[key] = (qubit_indices[operation[key]])
                        else:
                            operation[key] = (qubit_indices)

        # Add the other circuit's log to the circuit log
        self.circuit_log.extend(circuit.circuit_log)

        # Create the updated circuit
        self.circuit = self.convert(type(self)).circuit

    @abstractmethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
        """ Measure the qubits in the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Collection[int]
            The indices of the qubits to measure.

        Usage
        -----
        >>> circuit.measure(qubit_indices=0)
        >>> circuit.measure(qubit_indices=[0, 1])
        """

    @abstractmethod
    def get_statevector(self,
                        backend: Backend | None = None) -> Collection[float]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `backend` : qickit.backend.Backend, optional
            The backend to run the circuit on.

        Returns
        -------
        `statevector` : Collection[float]
            The statevector of the circuit.

        Usage
        -----
        >>> circuit.get_statevector()
        >>> circuit.get_statevector(backend=backend)
        """

    @abstractmethod
    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict[str, int]:
        """ Get the counts of the circuit.

        Parameters
        ----------
        `num_shots` : int
            The number of shots to run.
        `backend` : qickit.backend.Backend, optional
            The backend to run the circuit on.

        Returns
        -------
        `counts` : dict[str, int]
            The counts of the circuit.

        Usage
        -----
        >>> circuit.get_counts(num_shots=1024)
        >>> circuit.get_counts(num_shots=1024, backend=backend)
        """

    @abstractmethod
    def get_depth(self) -> int:
        """ Get the depth of the circuit.

        Returns
        -------
        `depth` : int
            The depth of the circuit.

        Usage
        -----
        >>> circuit.get_depth()
        """

    def get_width(self) -> int:
        """ Get the width of the circuit.

        Returns
        -------
        `width` : int
            The width of the circuit.

        Usage
        -----
        >>> circuit.get_width()
        """
        return self.num_qubits

    @abstractmethod
    def get_unitary(self) -> NDArray[np.number]:
        """ Get the unitary matrix of the circuit.

        Returns
        -------
        `unitary` : NDArray[np.number]
            The unitary matrix of the circuit.

        Usage
        -----
        >>> circuit.get_unitary()
        """

    @abstractmethod
    def transpile(self) -> None:
        """ Transpile the circuit to U3 and CX gates.

        Usage
        -----
        >>> circuit.transpile()
        """

    def compress(self,
                 compression_percentage: float) -> None:
        """ Compresses the circuit angles.

        Parameters
        ----------
        `compression_percentage` : float
            The percentage of compression. Value between 0.0 to 1.0.

        Usage
        -----
        >>> circuit.compress(compression_percentage=0.1)
        """
        if not 0 <= compression_percentage <= 1:
            raise ValueError("The compression percentage must be between 0 and 1.")
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
                       qubit_indices: Collection[int]) -> None:
        """ Change the mapping of the circuit.

        Parameters
        ----------
        `qubit_indices` : Collection[int]
            The updated order of the qubits.

        Raises
        ------
        TypeError
            - Qubit indices must be a collection.
            - All qubit indices must be integers.
        ValueError
            The number of qubits must match the number of qubits in the circuit.

        Usage
        -----
        >>> circuit.change_mapping(qubit_indices=[1, 0])
        """
        # Convert the qubit indices to a list if it is a range
        if isinstance(qubit_indices, range):
            qubit_indices = list(qubit_indices)

        # Check if the qubit indices are a collection
        if not isinstance(qubit_indices, Collection):
            raise TypeError("Qubit indices must be a collection.")

        # Check if all qubit indices are integers
        if not all(isinstance(index, int) for index in qubit_indices):
            raise TypeError("All qubit indices must be integers.")

        # The number of qubits must match the number of qubits in the circuit.
        if self.num_qubits != len(qubit_indices):
            raise ValueError("The number of qubits must match the number of qubits in the circuit.")

        # Update the qubit indices
        for operation in self.circuit_log:
            keys = QUBIT_KEYS.union(QUBIT_LIST_KEYS)
            for key in keys:
                if key in operation:
                    if isinstance(operation[key], Collection):
                        operation[key] = [(qubit_indices[index]) for index in operation[key]]
                    else:
                        operation[key] = (qubit_indices[operation[key]])

        # Convert the circuit to create the updated circuit
        self.circuit = self.convert(type(self)).circuit

    def convert(self,
                circuit_framework: Type[Circuit]) -> Circuit:
        """ Convert the circuit to another circuit framework.

        Parameters
        ----------
        `circuit_framework` : qickit.circuit.Circuit
            The circuit framework to convert to.

        Returns
        -------
        `converted_circuit` : qickit.circuit.Circuit
            The converted circuit.

        Usage
        -----
        >>> circuit.convert(circuit_framework=QiskitCircuit)
        """
        # Define the new circuit using the provided framework
        converted_circuit = circuit_framework(self.num_qubits, self.num_clbits)

        # Define a mapping between Qiskit gate names and corresponding methods in the target framework
        gate_mapping: dict[str, Callable] = {
            'Identity': converted_circuit.Identity,
            'X': converted_circuit.X,
            'Y': converted_circuit.Y,
            'Z': converted_circuit.Z,
            'H': converted_circuit.H,
            'S': converted_circuit.S,
            'T': converted_circuit.T,
            'RX': converted_circuit.RX,
            'RY': converted_circuit.RY,
            'RZ': converted_circuit.RZ,
            'U3': converted_circuit.U3,
            'SWAP': converted_circuit.SWAP,
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
            'CSWAP': converted_circuit.CSWAP,
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
            'MCSWAP': converted_circuit.MCSWAP,
            'GlobalPhase': converted_circuit.GlobalPhase,
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

        return converted_circuit

    @abstractmethod
    def to_qasm(self) -> str:
        """ Convert the circuit to QASM.

        Returns
        -------
        `qasm` : str
            The QASM representation of the circuit.

        Usage
        -----
        >>> circuit.to_qasm()
        """

    @staticmethod
    def from_cirq(cirq_circuit: cirq.Circuit,
                  output_framework: Type[Circuit]) -> Circuit:
        """ Create a `qickit.Circuit` from a `cirq.Circuit`.

        Parameters
        ----------
        `cirq_circuit` : cirq.Circuit
            The Cirq quantum circuit to convert.
        `output_framework` : qickit.circuit.Circuit
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Usage
        -----
        >>> circuit.from_cirq(cirq_circuit)
        """
        # Define a circuit
        num_qubits = len(cirq_circuit.all_qubits())
        circuit = output_framework(num_qubits=num_qubits, num_clbits=num_qubits)

        # Define the list of all circuit operations
        ops = list(cirq_circuit.all_operations())

        # Iterate over the operations in the Cirq circuit
        for operation in ops:
            # Extract the gate type
            gate = operation.gate
            gate_type = type(gate).__name__

            # Extract the qubit indices
            qubits = operation.qubits
            qubit_indices = [qubit.x for qubit in qubits] if len(qubits) > 1 else qubits[0].x # type: ignore

            # Extract the parameters of the gate
            parameters = gate._json_dict_() # type: ignore

            # TODO: Add U3, CU3, and MCU3 support (Note: Cirq doesn't have built-in U3 gate)
            # TODO: Add GlobalPhase gate support (Note: Cirq doesn't have global phase attribute)
            if gate_type == 'IdentityGate':
                circuit.Identity(qubit_indices)

            elif gate_type == '_PauliX':
                circuit.X(qubit_indices)

            elif gate_type == '_PauliY':
                circuit.Y(qubit_indices)

            elif gate_type == '_PauliZ':
                circuit.Z(qubit_indices)

            elif gate_type == 'HPowGate':
                circuit.H(qubit_indices)

            elif gate_type == 'ZPowGate':
                if parameters['exponent'] == 0.5:
                    circuit.S(qubit_indices)
                elif parameters['exponent'] == 0.25:
                    circuit.T(qubit_indices)

            elif gate_type == 'Rx':
                if isinstance(qubit_indices, list):
                    for qubit_index in qubit_indices:
                        circuit.RX(parameters['rads'], qubit_index)
                else:
                    circuit.RX(parameters['rads'], qubit_indices)

            elif gate_type == 'Ry':
                if isinstance(qubit_indices, list):
                    for qubit_index in qubit_indices:
                        circuit.RY(parameters['rads'], qubit_index)
                else:
                    circuit.RY(parameters['rads'], qubit_indices)

            elif gate_type == 'Rz':
                if isinstance(qubit_indices, list):
                    for qubit_index in qubit_indices:
                        circuit.RZ(parameters['rads'], qubit_index)
                else:
                    circuit.RZ(parameters['rads'], qubit_indices)

            elif gate_type == 'SWAP':
                circuit.SWAP(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'ControlledGate':
                if parameters['sub_gate'] == cirq.X:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCX(control_indices=qubit_indices[:-1],
                                    target_indices=qubit_indices[-1])
                    else:
                        circuit.CX(qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.Y:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCY(control_indices=qubit_indices[:-1],
                                    target_indices=qubit_indices[-1])
                    else:
                        circuit.CY(qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.Z:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCZ(control_indices=qubit_indices[:-1],
                                    target_indices=qubit_indices[-1])
                    else:
                        circuit.CZ(qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.H:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCH(control_indices=qubit_indices[:-1],
                                    target_indices=qubit_indices[-1])
                    else:
                        circuit.CH(qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.S:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCS(control_indices=qubit_indices[:-1],
                                    target_indices=qubit_indices[-1])
                    else:
                        circuit.CS(qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.T:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCT(control_indices=qubit_indices[:-1],
                                    target_indices=qubit_indices[-1])
                    else:
                        circuit.CT(qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.Rx:
                    angle = parameters['sub_gate']._json_dict_()['rads']
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCRX(angle,
                                     control_indices=qubit_indices[:-1],
                                     target_indices=qubit_indices[-1])
                    else:
                        circuit.CRX(angle, qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.Ry:
                    angle = parameters['sub_gate']._json_dict_()['rads']
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCRY(angle,
                                     control_indices=qubit_indices[:-1],
                                     target_indices=qubit_indices[-1])
                    else:
                        circuit.CRY(angle, qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.Rz:
                    angle = parameters['sub_gate']._json_dict_()['rads']
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCRZ(angle,
                                     control_indices=qubit_indices[:-1],
                                     target_indices=qubit_indices[-1])
                    else:
                        circuit.CRZ(angle, qubit_indices[0], qubit_indices[1])

                elif parameters['sub_gate'] == cirq.SWAP:
                    if len(parameters['control_qid_shape']) > 1:
                        circuit.MCSWAP(control_indices=qubit_indices[:-2],
                                       first_target_index=qubit_indices[-2],
                                       second_target_index=qubit_indices[-1])
                    else:
                        circuit.CSWAP(qubit_indices[0], qubit_indices[1], qubit_indices[2])

            else:
                raise ValueError(f"Gate not supported.\n{operation} ")

        return circuit

    @staticmethod
    def from_pennylane(pennylane_circuit: qml.QNode,
                       output_framework: Type[Circuit]) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `qml.QNode`.

        Parameters
        ----------
        `pennylane_circuit` : qml.QNode
            The PennyLane quantum circuit to convert.
        `output_framework` : qickit.circuit.Circuit
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Usage
        -----
        >>> circuit.from_pennylane(pennylane_circuit)
        """
        # Define a circuit
        num_qubits = len(pennylane_circuit.device.wires)
        circuit = output_framework(num_qubits=num_qubits, num_clbits=num_qubits)

        # TODO: Implement the conversion from PennyLane to Qickit
        return circuit

    @staticmethod
    def from_qiskit(qiskit_circuit: qiskit.QuantumCircuit,
                    output_framework: Type[Circuit]) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `qiskit.QuantumCircuit`.

        Parameters
        ----------
        `qiskit_circuit` : qiskit.QuantumCircuit
            The Qiskit quantum circuit to convert.
        `output_framework` : qickit.circuit.Circuit
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Usage
        -----
        >>> circuit.from_qiskit(qiskit_circuit)
        """
        def match_pattern(string: str,
                          gate_name: str) -> bool:
            """ Check if the string matches the pattern.

            Parameters
            ----------
            `string` : str
                The string to check.
            `gate_name` : str
                The name of the gate.

            Returns
            -------
            `match` : bool
                Whether or not the string matches the pattern.
            """
            pattern = re.compile(fr'c\d+{gate_name}')
            if pattern.match(string):
                return True
            return False

        # Define a circuit
        num_qubits = qiskit_circuit.num_qubits
        circuit = output_framework(num_qubits=num_qubits, num_clbits=num_qubits)

        # Iterate over the operations in the Qiskit circuit
        # TODO: Add Identity gate
        for gate in qiskit_circuit.data:
            # Extract the gate type
            gate_type = gate[0].name

            # Extract the qubit indices
            qubit_indices = [qubit._index for qubit in gate[1]] if len(gate[1]) > 1 else gate[1][0]._index

            if gate_type == 'id':
                circuit.Identity(qubit_indices)

            elif gate_type == 'x':
                circuit.X(qubit_indices)

            elif gate_type == 'y':
                circuit.Y(qubit_indices)

            elif gate_type == 'z':
                circuit.Z(qubit_indices)

            elif gate_type == 'h':
                circuit.H(qubit_indices)

            elif gate_type == 's':
                circuit.S(qubit_indices)

            elif gate_type == 't':
                circuit.T(qubit_indices)

            elif gate_type == 'rx':
                circuit.RX(gate[0].params[0], qubit_indices)

            elif gate_type == 'ry':
                circuit.RY(gate[0].params[0], qubit_indices)

            elif gate_type == 'rz':
                circuit.RZ(gate[0].params[0], qubit_indices)

            elif gate_type == 'u3':
                circuit.U3(gate[0].params, qubit_indices)

            elif gate_type == 'swap':
                circuit.SWAP(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cx':
                circuit.CX(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cy':
                circuit.CY(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cz':
                circuit.CZ(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'ch':
                circuit.CH(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cs':
                circuit.CS(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'ct':
                circuit.CT(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'crx':
                circuit.CRX(gate[0].params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cry':
                circuit.CRY(gate[0].params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == 'crz':
                circuit.CRZ(gate[0].params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cu3':
                circuit.CU3(gate[0].params, qubit_indices[0], qubit_indices[1])

            elif gate_type == 'cswap':
                circuit.CSWAP(qubit_indices[0], qubit_indices[1], qubit_indices[2])

            elif gate_type == 'mcx' or gate_type == 'ccx':
                circuit.MCX(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'y'):
                circuit.MCY(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'z'):
                circuit.MCZ(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'h'):
                circuit.MCH(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 's'):
                circuit.MCS(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 't'):
                circuit.MCT(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'rx'):
                circuit.MCRX(gate[0].params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'ry'):
                circuit.MCRY(gate[0].params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'rz'):
                circuit.MCRZ(gate[0].params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, 'u3'):
                circuit.MCU3(gate[0].params, qubit_indices[:-1], qubit_indices[-1])

            else:
                raise ValueError(f"Gate not supported.\n{gate_type} ")

        # Apply the global phase of the `qiskit_circuit`
        circuit.GlobalPhase(qiskit_circuit.global_phase)

        return circuit

    @staticmethod
    def from_tket(tket_circuit: pytket.Circuit,
                  output_framework: Type[Circuit]) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `tket.Circuit`.

        Parameters
        ----------
        `tket_circuit` : tket.Circuit
            The TKET quantum circuit to convert.
        `output_framework` : qickit.circuit.Circuit
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Usage
        -----
        >>> circuit.from_tket(tket_circuit)
        """
        # Define a circuit
        num_qubits = tket_circuit.n_qubits
        circuit = output_framework(num_qubits=num_qubits, num_clbits=num_qubits)

        # Iterate over the operations in the Qiskit circuit
        for gate in tket_circuit:
            # Extract the gate type
            gate_type = str(gate.op.type)

            # Extract the qubit indices
            qubit_indices = [qubit.index[0] for qubit in gate.qubits] if len(gate.qubits) > 1 \
                                                                      else gate.qubits[0].index[0]

            if gate_type == 'OpType.I':
                circuit.Identity(qubit_indices)

            elif gate_type == 'OpType.X':
                circuit.X(qubit_indices)

            elif gate_type == 'OpType.Y':
                circuit.Y(qubit_indices)

            elif gate_type == 'OpType.Z':
                circuit.Z(qubit_indices)

            elif gate_type == 'OpType.H':
                circuit.H(qubit_indices)

            elif gate_type == 'OpType.S':
                circuit.S(qubit_indices)

            elif gate_type == 'OpType.T':
                circuit.T(qubit_indices)

            elif gate_type == 'OpType.Rx':
                circuit.RX(float(gate.op.params[0]), qubit_indices)

            elif gate_type == 'OpType.Ry':
                circuit.RY(float(gate.op.params[0]), qubit_indices)

            elif gate_type == 'OpType.Rz':
                circuit.RZ(float(gate.op.params[0]), qubit_indices)

            elif gate_type == 'OpType.U3':
                circuit.U3([float(param) for param in gate.op.params], qubit_indices)

            elif gate_type == 'OpType.SWAP':
                circuit.SWAP(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CX':
                circuit.CX(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CY':
                circuit.CY(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CZ':
                circuit.CZ(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CH':
                circuit.CH(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CS':
                circuit.CS(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CT':
                circuit.CT(qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CRx':
                circuit.CRX(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CRy':
                circuit.CRY(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CRz':
                circuit.CRZ(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CU3':
                circuit.CU3([float(param) for param in gate.op.params], qubit_indices[0], qubit_indices[1])

            elif gate_type == 'OpType.CnX':
                circuit.MCX(qubit_indices[:-1], qubit_indices[-1])

            elif gate_type == 'OpType.CnY':
                circuit.MCY(qubit_indices[:-1], qubit_indices[-1])

            elif gate_type == 'OpType.CnZ':
                circuit.MCZ(qubit_indices[:-1], qubit_indices[-1])

            elif gate_type == 'OpType.QControlBox':
                qcontrolbox: pytket.circuit.QControlBox = gate.op

                if str(qcontrolbox.get_op()) == 'X':
                    circuit.MCX(qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'Y':
                    circuit.MCY(qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'Z':
                    circuit.MCZ(qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'H':
                    circuit.MCH(qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'S':
                    circuit.MCS(qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'T':
                    circuit.MCT(qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'Rx':
                    circuit.MCRX(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'Ry':
                    circuit.MCRY(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'Rz':
                    circuit.MCRZ(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'U3':
                    circuit.MCU3([float(param) for param in gate.op.get_op().params], qubit_indices[:-1], qubit_indices[-1])

                elif str(qcontrolbox.get_op()) == 'SWAP':
                    circuit.MCSWAP(qubit_indices[:-2], qubit_indices[-2], qubit_indices[-1])

            else:
                raise ValueError(f"Gate not supported.\n{gate_type} ")

        # Apply the global phase of the `tket_circuit`
        circuit.GlobalPhase(tket_circuit.phase/np.pi)

        return circuit

    @staticmethod
    def from_qasm(qasm: str,
                  output_framework: Type[Circuit]) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a QASM string.

        Parameters
        ----------
        `qasm` : str
            The QASM string to convert.
        `output_framework` : qickit.circuit.Circuit
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Usage
        -----
        >>> circuit.from_qasm(qasm)
        """
        # Define a circuit
        num_qubits = 0
        circuit = output_framework(num_qubits=num_qubits, num_clbits=num_qubits)

        # TODO: Implement the conversion from QASM to Qickit
        return circuit

    def reset(self) -> None:
        """ Reset the circuit to an empty circuit.

        Usage
        -----
        >>> circuit.reset()
        """
        self.circuit_log = []
        self.circuit = type(self)(self.num_qubits, self.num_clbits).circuit

    @abstractmethod
    def draw(self):
        """ Draw the circuit.

        Usage
        -----
        >>> circuit.draw()
        """

    def plot_histogram(self,
                       non_zeros_only: bool = False) -> plt.Figure:
        """ Plot the histogram of the circuit.

        Parameters
        ----------
        `non_zeros_only` : bool
            Whether or not to plot only the non-zero counts.

        Returns
        -------
        `figure` : matplotlib.pyplot.Figure
            The figure of the histogram.

        Usage
        -----
        >>> circuit.plot_histogram()
        >>> circuit.plot_histogram(non_zeros_only=True)
        """
        # Get the counts of the circuit
        counts = self.get_counts(1024)

        if non_zeros_only:
            # Remove the zero counts
            counts = {key: value for key, value in counts.items() if value != 0}

        # Plot the histogram
        figure = plt.figure()
        plt.bar(counts.keys(), counts.values(), 0.5) # type: ignore
        plt.xlabel('State')
        plt.ylabel('Counts')
        plt.title('Histogram of the Circuit')
        plt.close()

        return figure

    def __eq__(self,
               other_circuit: object) -> bool:
        """ Compare two circuits for equality.

        Parameters
        ----------
        `other_circuit` : object
            The other circuit to compare to.

        Returns
        -------
        bool
            Whether the two circuits are equal.

        Raises
        ------
        TypeError
            Circuits must be compared with other circuits.
        """
        if not isinstance(other_circuit, Circuit):
            raise TypeError("Circuits must be compared with other circuits.")

        print(self.circuit_log == other_circuit.circuit_log)

        return self.circuit_log == other_circuit.circuit_log

    def __len__(self) -> int:
        """ Get the number of the circuit operations.

        Returns
        -------
        int
            The number of the circuit operations.
        """
        return len(self.circuit_log)

    def __str__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        str
            The string representation of the circuit.
        """
        return str(self.circuit_log)

    def __repr__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        str
            The string representation of the circuit.
        """
        return f"Circuit(num_qubits={self.num_qubits}, num_clbits={self.num_clbits})"

    @classmethod
    def __subclasscheck__(cls, C) -> bool:
        """ Checks if a class is a `qickit.circuit.Circuit` if the class
        passed does not directly inherit from `qickit.circuit.Circuit`.

        Parameters
        ----------
        `C` : type
            The class to check if it is a subclass.

        Returns
        -------
        bool
            Whether or not the class is a subclass.
        """
        if cls is Circuit:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return False

    @classmethod
    def __subclasshook__(cls, C) -> bool | NotImplementedType:
        """ Checks if a class is a `qickit.circuit.Circuit` if the class
        passed does not directly inherit from `qickit.circuit.Circuit`.

        Parameters
        ----------
        `C` : type
            The class to check if it is a subclass.

        Returns
        -------
        bool | NotImplementedType
            Whether or not the class is a subclass.
        """
        if cls is Circuit:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return NotImplemented

    @classmethod
    def __instancecheck__(cls, C) -> bool:
        """ Checks if an object is a `qickit.circuit.Circuit` given its
        interface.

        Parameters
        ----------
        `C` : object
            The instance to check.

        Returns
        -------
        bool
            Whether or not the instance is a `qickit.circuit.Circuit`.
        """
        if cls is Circuit:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return False