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

""" Abstract base class for creating and manipulating gate-based circuits.
"""

from __future__ import annotations

__all__ = ["Circuit"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
import copy
import cmath
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from numpy.typing import NDArray
from types import NotImplementedType
from typing import (
    Any, Callable, Literal, overload, SupportsFloat, SupportsIndex, Type, TYPE_CHECKING
)

import qiskit # type: ignore
import cirq # type: ignore
import pennylane as qml # type: ignore
import pytket
import pytket.circuit
import quimb.tensor as qtn # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit.circuit_utils import (
    extract_rz, decompose_uc_rotations, decompose_ucg_help, simplify
)
from qickit.circuit.dag import DAGCircuit
from qickit.circuit.from_framework import FromCirq, FromQiskit, FromTKET
from qickit.predicates import is_unitary_matrix
from qickit.primitives import Bra, Ket, Operator
from qickit.synthesis.gate_decompositions.multi_controlled_decomposition import MCRX, MCRY, MCRZ
from qickit.synthesis.statepreparation import Isometry
from qickit.synthesis.unitarypreparation import (
    UnitaryPreparation, ShannonDecomposition, QiskitUnitaryTranspiler
)

EPSILON = 1e-10

""" Set the frozensets for the keys to be used:
- Decorator `Circuit.gatemethod()`
- Method `Circuit.vertical_reverse()`
- Method `Circuit.horizontal_reverse()`
- Method `Circuit.add()`
- Method `Circuit.change_mapping()`
"""
QUBIT_KEYS = frozenset([
    "qubit_index", "control_index", "target_index", "first_qubit_index",
    "second_qubit_index", "first_target_index", "second_target_index"
])
QUBIT_LIST_KEYS = frozenset(["qubit_indices", "control_indices", "target_indices"])
ANGLE_KEYS = frozenset(["angle", "angles"])
ALL_QUBIT_KEYS = QUBIT_KEYS.union(QUBIT_LIST_KEYS)

# Define the mapping for controlled operation
CONTROL_MAPPING = {
    "qubit_indices": "target_indices",
    "control_index": "control_indices",
    "control_indices": "control_indices",
    "target_index": "target_indices",
    "target_indices": "target_indices",
    "first_qubit_index": "first_target_index",
    "second_qubit_index": "second_target_index",
    "first_target_index": "first_target_index",
    "second_target_index": "second_target_index"
}

# List of 1Q gates wrapped by individual frameworks
GATES = Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase", "U3"]


class Circuit(ABC):
    """ `qickit.circuit.Circuit` is the class for creating and manipulating gate-based circuits.
    This class is defined for external Quantum Circuit (QC) Frameworks.
    Current supported packages are :
    - IBM Qiskit
    - Google's Cirq
    - NVIDIA's CUDA-Quantum
    - Quantinuum's PyTKET
    - Xanadu's PennyLane

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `circuit` : Circuit_Type
        The circuit framework type.
    `gate_mapping` : dict[str, Callable]
        The mapping of the gates to the circuit.
    `measured_qubits` : set[int]
        The set of measured qubits indices.
    `circuit_log` : list[dict]
        The log of the circuit operations.
    `stack` : list[list[dict]]
        The stack of the circuit log. Used for logging gate definitions.
    `global_phase` : float
        The global phase of the circuit.
    `process_gate_params_flag` : bool
        The flag to process the gate parameters.

    Raises
    ------
    TypeError
        - Number of qubits must be integers.
    ValueError
        - Number of qubits must be greater than 0.
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:
        """ Initialize a `qickit.circuit.Circuit` instance.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("Number of qubits must be integers.")

        if num_qubits < 1:
            raise ValueError("Number of qubits must be greater than 0.")

        self.num_qubits = num_qubits
        self.circuit: Any
        self.gate_mapping: dict[str, Callable] = self._define_gate_mapping()
        self.measured_qubits: set[int] = set()
        self.circuit_log: list[dict] = []
        self.stack: list[list[dict]] = [self.circuit_log]
        self.global_phase: float = 0
        self.process_gate_params_flag: bool = True

    def _convert_param_type(
            self,
            value: Any
        ) -> int | float | list:
        """ Convert parameter types for consistency.

        Parameters
        ----------
        `value` : Any
            The value to convert.

        Returns
        -------
        `value` : int | float | list
            The converted value.
        """
        if isinstance(value, (range, tuple, Sequence)):
            value = list(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, SupportsIndex):
            value = int(value)
        elif isinstance(value, SupportsFloat):
            value = float(value)
        return value

    def _validate_qubit_index(
            self,
            name: str,
            value: Any
        ) -> int | list[int]:
        """ Validate qubit indices are within the valid range.

        Parameters
        ----------
        `name` : str
            The name of the parameter.
        `value` : Any
            The value of the parameter.

        Returns
        -------
        `value` : int | list[int]
            The value of the parameter.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        IndexError
            - Qubit index out of range.
        """
        if name in ALL_QUBIT_KEYS:
            if isinstance(value, list):
                if len(value) == 1:
                    value = value[0]

                    if not isinstance(value, int):
                        raise TypeError(f"Qubit index must be an integer. Unexpected type {type(value)} received.")

                    if value >= self.num_qubits or value < -self.num_qubits:
                        raise IndexError(f"Qubit index {value} out of range {self.num_qubits-1}.")

                    value = value if value >= 0 else self.num_qubits + value

                else:
                    for i, index in enumerate(value):
                        if not isinstance(index, int):
                            raise TypeError(f"Qubit index must be an integer. Unexpected type {type(value)} received.")

                        if index >= self.num_qubits or index < -self.num_qubits:
                            raise IndexError(f"Qubit index {index} out of range {self.num_qubits-1}.")

                        value[i] = index if index >= 0 else self.num_qubits + index

            elif isinstance(value, int):
                if value >= self.num_qubits or value < -self.num_qubits:
                    raise IndexError(f"Qubit index {value} out of range {self.num_qubits-1}.")

                value = value if value >= 0 else self.num_qubits + value

        return value

    def _validate_angle(
            self,
            name: str,
            value: Any
        ) -> None | float | list[float]:
        """ Ensure angles are valid and not effectively zero.

        Parameters
        ----------
        `name` : str
            The name of the parameter.
        `value` : Any
            The value of the parameter.

        Returns
        -------
        `value` : None | float | list[float]
            The value of the parameter. If the value is effectively zero, return None.
            This is to indicate that no operation is needed.

        Raises
        ------
        TypeError
            - Angle must be a number.
        """
        if name in ANGLE_KEYS:
            if isinstance(value, list):
                for angle in value:
                    if not isinstance(angle, (int, float)):
                        raise TypeError(f"Angle must be a number. Unexpected type {type(angle)} received.")
                    if abs(angle) <= EPSILON or abs(angle % (2 * np.pi)) <= EPSILON:
                        angle = 0

                if all(angle == 0 for angle in value):
                    # Indicate no operation needed
                    return None
            else:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Angle must be a number. Unexpected type {type(value)} received.")
                if abs(value) <= EPSILON or abs(value % (2 * np.pi)) <= EPSILON:
                    # Indicate no operation needed
                    return None

        return value

    def process_gate_params(
            self,
            gate: str,
            params: dict
        ) -> dict | None:
        """ Process the gate parameters for the circuit.

        Parameters
        ----------
        `gate` : str
            The gate to apply to the circuit.
        `params` : dict
            The parameters of the gate.

        Returns
        -------
        `gate_dict` : dict | None
            The dictionary of the gate parameters. If no
            operation is needed, return None.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        IndexError
            - Qubit index out of range.

        Usage
        -----
        >>> gate = self.process_gate_params(gate="X", params={"qubit_indices": 0})
        """
        if not self.process_gate_params_flag:
            return None

        # Remove the "self" key from the dictionary to avoid the inclusion of str(circuit)
        # in the circuit log
        params.pop("self", None)

        for name, value in params.items():
            value = self._convert_param_type(value)
            value = self._validate_qubit_index(name, value)

            if value is None:
                continue

            value = self._validate_angle(name, value)

            # Indicate no operation needed
            if value is None:
                return None

            params[name] = value

        # Add the gate to the circuit log
        gate_dict = {"gate": gate, **params, "definition": []}
        self.stack[-1].append(gate_dict)

        return gate_dict

    @contextmanager
    def decompose_last(
            self,
            gate: dict | None
        ):
        """ Decompose the last gate in the circuit.

        Notes
        -----
        This context manager is used to define new gates by labelling a sequence
        of operations as a new gate. This is useful for defining new gates in the
        circuit without introducing breaking changes.

        Usage
        -----
        >>> def NewGate(qubit_indices: int | Sequence[int]):
        >>>     gate = self.process_gate_params(gate="NewGate", params=locals())
        >>>     with circuit.decompose_last():
        >>>         circuit.X(qubit_indices)
        """
        # If the gate is parameterized, and its rotation is effectively zero, return
        # as no operation is needed
        if gate is None:
            yield
            return

        self.stack.append(gate["definition"])

        try:
            yield
        finally:
            self.stack.pop()

    @staticmethod
    @abstractmethod
    def _define_gate_mapping() -> dict[str, Callable]:
        """ Define the gate mapping for the circuit.

        Notes
        -----
        The gate mapping is defined for each QC framework, and is meant to be used internally.

        Returns
        -------
        `gate_mapping` : dict[str, Callable]
            The mapping of the gates to the circuit.
        """

    @abstractmethod
    def _gate_mapping(
            self,
            gate: GATES,
            target_indices: int | Sequence[int],
            control_indices: int | Sequence[int] = [],
            angles: Sequence[float] = [0, 0, 0]
        ) -> None:
        """ Apply a gate to the circuit.

        Notes
        -----
        The gate mapping is defined for each QC framework, and is meant to be used internally.

        Parameters
        ----------
        `gate` : GATES
            The gate to apply to the circuit.
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).
        `control_indices` : int | Sequence[int], optional, default=[]
            The index of the control qubit(s).
        `angles` : Sequence[float], optional, default=[0, 0, 0]
            The rotation angles in radians.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Gate not supported.
            - Qubit index out of range.

        Usage
        -----
        >>> circuit._gate_mapping(gate="X", target_indices=0)
        >>> circuit._gate_mapping(gate="X", target_indices=[0, 1])
        >>> circuit._gate_mapping(gate="RX", target_indices=0, angles=[np.pi/2])
        >>> circuit._gate_mapping(gate="RX", target_indices=[0, 1], angles=[np.pi/2])
        >>> circuit._gate_mapping(gate="U3", target_indices=0, angles=[np.pi/2, np.pi/2, np.pi/2])
        >>> circuit._gate_mapping(gate="U3", target_indices=[0, 1], angles=[np.pi/2, np.pi/2, np.pi/2])
        >>> circuit._gate_mapping(gate="MCX", target_indices=2, control_indices=[0, 1])
        >>> circuit._gate_mapping(gate="MCX", target_indices=[2, 3], control_indices=[0, 1])
        >>> circuit._gate_mapping(gate="MCRX", target_indices=2, control_indices=[0, 1], angles=[np.pi/2])
        >>> circuit._gate_mapping(gate="MCRX", target_indices=[2, 3], control_indices=[0, 1], angles=[np.pi/2])
        >>> circuit._gate_mapping(gate="MCU3", target_indices=2, control_indices=[0, 1], angles=[np.pi/2, np.pi/2, np.pi/2])
        >>> circuit._gate_mapping(gate="MCU3", target_indices=[2, 3], control_indices=[0, 1], angles=[np.pi/2, np.pi/2, np.pi/2])
        """

    def Identity(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply an Identity gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.Identity(qubit_indices=0)
        >>> circuit.Identity(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.Identity.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([0, 0, 0], qubit_indices)

    def X(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Pauli-X gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.X(qubit_indices=0)
        >>> circuit.X(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.X.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([np.pi, 0, np.pi], qubit_indices)

    def Y(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Pauli-Y gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.Y(qubit_indices=0)
        >>> circuit.Y(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.Y.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([np.pi, np.pi/2, np.pi/2], qubit_indices)

    def Z(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Pauli-Z gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.Z(qubit_indices=0)
        >>> circuit.Z(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.Z.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(np.pi, qubit_indices)

    def H(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Hadamard gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.H(qubit_indices=0)
        >>> circuit.H(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.H.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([np.pi/2, 0, np.pi], qubit_indices)

    def S(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Clifford-S gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.S(qubit_indices=0)
        >>> circuit.S(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.S.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(np.pi/2, qubit_indices)

    def Sdg(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Clifford-S^{dagger} gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.Sdg(qubit_indices=0)
        >>> circuit.Sdg(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.Sdg.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(-np.pi/2, qubit_indices)

    def T(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Clifford-T gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.T(qubit_indices=0)
        >>> circuit.T(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.T.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(np.pi/4, qubit_indices)

    def Tdg(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Clifford-T^{dagger} gate to the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.Tdg(qubit_indices=0)
        >>> circuit.Tdg(qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.Tdg.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(-np.pi/4, qubit_indices)

    def RX(
            self,
            angle: float,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a RX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.RX(angle=np.pi/2, qubit_indices=0)
        >>> circuit.RX(angle=np.pi/2, qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.RX.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([angle, -np.pi/2, np.pi/2], qubit_indices)

    def RY(
            self,
            angle: float,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a RY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.RY(angle=np.pi/2, qubit_index=0)
        >>> circuit.RY(angle=np.pi/2, qubit_index=[0, 1])
        """
        gate = self.process_gate_params(gate=self.RY.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([angle, 0, 0], qubit_indices)

    def RZ(
            self,
            angle: float,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a RZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.RZ(angle=np.pi/2, qubit_indices=0)
        >>> circuit.RZ(angle=np.pi/2, qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.RZ.__name__, params=locals())

        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        with self.decompose_last(gate):
            self.RX(np.pi/2, qubit_indices)
            self.RY(-angle, qubit_indices)
            self.RX(-np.pi/2, qubit_indices)

    def Phase(
            self,
            angle: float,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Phase gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.Phase(angle=np.pi/2, qubit_index=0)
        """
        gate = self.process_gate_params(gate=self.Phase.__name__, params=locals())

        with self.decompose_last(gate):
            self.U3([0, 0, angle], qubit_indices)

    def XPow(
            self,
            power: float,
            global_shift: float,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a X^power gate to the circuit.

        Notes
        -----
        The XPow gate is defined exactly as Google Cirq's `XPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.XPow(power=0.5, qubit_indices=0)
        >>> circuit.XPow(power=0.5, qubit_indices=[0, 1])
        >>> circuit.XPow(power=0.5, global_shift=0.5, qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.XPow.__name__, params=locals())

        with self.decompose_last(gate):
            qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

            self.RX(power * np.pi, qubit_indices)
            self.GlobalPhase(power * np.pi * (global_shift + 0.5) * len(qubit_indices))

    def YPow(
            self,
            power: float,
            global_shift: float,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Y^power gate to the circuit.

        Notes
        -----
        The YPow gate is defined exactly as Google Cirq's `YPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.YPow(power=0.5, qubit_indices=0)
        >>> circuit.YPow(power=0.5, qubit_indices=[0, 1])
        >>> circuit.YPow(power=0.5, global_shift=0.5, qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.YPow.__name__, params=locals())

        with self.decompose_last(gate):
            qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

            self.RY(power * np.pi, qubit_indices)
            self.GlobalPhase(power * np.pi * (global_shift + 0.5) * len(qubit_indices))

    def ZPow(
            self,
            power: float,
            global_shift: float,
            qubit_indices: int | Sequence[int],
        ) -> None:
        """ Apply a Z^power gate to the circuit.

        Notes
        -----
        The ZPow gate is defined exactly as Google Cirq's `ZPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.ZPow(power=0.5, qubit_indices=0)
        >>> circuit.ZPow(power=0.5, qubit_indices=[0, 1])
        >>> circuit.ZPow(power=0.5, global_shift=0.5, qubit_indices=[0, 1])
        """
        gate = self.process_gate_params(gate=self.ZPow.__name__, params=locals())

        with self.decompose_last(gate):
            qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

            self.RZ(power * np.pi, qubit_indices)
            self.GlobalPhase(power * np.pi * (global_shift + 0.5) * len(qubit_indices))

    def RXX(
            self,
            angle: float,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:
        """ Apply a RXX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `first_qubit_index` : int
            The index of the first qubit.
        `second_qubit_index` : int
            The index of the second qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.RXX(first_qubit_index=0, second_qubit_index=1, angle=np.pi/2)
        """
        gate = self.process_gate_params(gate=self.RXX.__name__, params=locals())

        with self.decompose_last(gate):
            self.H([first_qubit_index, second_qubit_index])
            self.CX(first_qubit_index, second_qubit_index)
            self.RZ(angle, second_qubit_index)
            self.CX(first_qubit_index, second_qubit_index)
            self.H([first_qubit_index, second_qubit_index])

    def RYY(
            self,
            angle: float,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:
        """ Apply a RYY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `first_qubit_index` : int
            The index of the first qubit.
        `second_qubit_index` : int
            The index of the second qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.RYY(first_qubit_index=0, second_qubit_index=1, angle=np.pi/2)
        """
        gate = self.process_gate_params(gate=self.RYY.__name__, params=locals())

        with self.decompose_last(gate):
            self.RX(np.pi/2, first_qubit_index)
            self.RX(np.pi/2, second_qubit_index)
            self.CX(first_qubit_index, second_qubit_index)
            self.RZ(angle, second_qubit_index)
            self.CX(first_qubit_index, second_qubit_index)
            self.RX(-np.pi/2, first_qubit_index)
            self.RX(-np.pi/2, second_qubit_index)

    def RZZ(
            self,
            angle: float,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:
        """ Apply a RZZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `first_qubit_index` : int
            The index of the first qubit.
        `second_qubit_index` : int
            The index of the second qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.RZZ(first_qubit_index=0, second_qubit_index=1, angle=np.pi/2)
        """
        gate = self.process_gate_params(gate=self.RZZ.__name__, params=locals())

        with self.decompose_last(gate):
            self.CX(first_qubit_index, second_qubit_index)
            self.RZ(angle, second_qubit_index)
            self.CX(first_qubit_index, second_qubit_index)

    def U3(
            self,
            angles: Sequence[float],
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `qubit_indices` : int
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.U3(angles=[np.pi/2, np.pi/2, np.pi/2], qubit_index=0)
        """
        _ = self.process_gate_params(gate=self.U3.__name__, params=locals())

        self._gate_mapping(
            gate="U3",
            angles=angles,
            target_indices=qubit_indices
        )

    def SWAP(
            self,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:
        """ Apply a SWAP gate to the circuit.

        Parameters
        ----------
        `first_qubit_index` : int
            The index of the first qubit.
        `second_qubit_index` : int
            The index of the second qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.SWAP(first_qubit_index=0, second_qubit_index=1)
        """
        gate = self.process_gate_params(gate=self.SWAP.__name__, params=locals())

        with self.decompose_last(gate):
            self.CX(first_qubit_index, second_qubit_index)
            self.CX(second_qubit_index, first_qubit_index)
            self.CX(first_qubit_index, second_qubit_index)

    def CX(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Pauli-X gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CX(control_index=0, target_index=1)
        """
        _ = self.process_gate_params(gate=self.CX.__name__, params=locals())

        self._gate_mapping(
            gate="X",
            control_indices=control_index,
            target_indices=target_index
        )

    def CY(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Pauli-Y gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CY(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CY.__name__, params=locals())

        with self.decompose_last(gate):
            self.Sdg(target_index)
            self.CX(control_index, target_index)
            self.S(target_index)

    def CZ(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Pauli-Z gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CZ(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CZ.__name__, params=locals())

        with self.decompose_last(gate):
            self.H(target_index)
            self.CX(control_index, target_index)
            self.H(target_index)

    def CH(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Hadamard gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CH(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CH.__name__, params=locals())

        with self.decompose_last(gate):
            self.S(target_index)
            self.H(target_index)
            self.T(target_index)
            self.CX(control_index, target_index)
            self.Tdg(target_index)
            self.H(target_index)
            self.Sdg(target_index)

    def CS(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Clifford-S gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CS(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CS.__name__, params=locals())

        with self.decompose_last(gate):
            self.T(control_index)
            self.CX(control_index, target_index)
            self.Tdg(target_index)
            self.CX(control_index, target_index)
            self.T(target_index)

    def CSdg(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Clifford-S^{dagger} gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CSdg(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CSdg.__name__, params=locals())

        with self.decompose_last(gate):
            self.Tdg(control_index)
            self.CX(control_index, target_index)
            self.T(target_index)
            self.CX(control_index, target_index)
            self.Tdg(target_index)

    def CT(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Clifford-T gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CT(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CT.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(np.pi/8, target_index)
            self.CX(control_index, target_index)
            self.Phase(-np.pi/8, target_index)
            self.CX(control_index, target_index)
            self.Phase(np.pi/8, control_index)

    def CTdg(
            self,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Clifford-T^{dagger} gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CTdg(control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CTdg.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(-np.pi/8, control_index)
            self.CX(control_index, target_index)
            self.Phase(np.pi/8, target_index)
            self.CX(control_index, target_index)
            self.Phase(-np.pi/8, target_index)

    def CRX(
            self,
            angle: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled RX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CRX(angle=np.pi/2, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CRX.__name__, params=locals())

        with self.decompose_last(gate):
            # Explicit optimal implementation covered in MCRX
            self.MCRX(angle, control_index, target_index)

    def CRY(
            self,
            angle: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled RY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CRY(angle=np.pi/2, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CRY.__name__, params=locals())

        with self.decompose_last(gate):
            # Explicit optimal implementation covered in MCRY
            self.MCRY(angle, control_index, target_index)

    def CRZ(
            self,
            angle: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled RZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CRZ(angle=np.pi/2, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CRZ.__name__, params=locals())

        with self.decompose_last(gate):
            # Explicit optimal implementation covered in MCRZ
            self.MCRZ(angle, control_index, target_index)

    def CPhase(
            self,
            angle: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Phase gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CPhase(angle=np.pi/2, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CPhase.__name__, params=locals())

        with self.decompose_last(gate):
            self.Phase(angle/2, control_index)
            self.CX(control_index, target_index)
            self.Phase(-angle/2, target_index)
            self.CX(control_index, target_index)
            self.Phase(angle/2, target_index)

    def CXPow(
            self,
            power: float,
            global_shift: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled X^power gate to the circuit.

        Notes
        -----
        The CXPow gate is defined exactly as Google Cirq's controlled `XPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CXPow(power=0.5, control_index=0, target_index=1)
        >>> circuit.CXPow(power=0.5, global_shift=0.5, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CXPow.__name__, params=locals())

        with self.decompose_last(gate):
            self.CRX(power * np.pi, control_index, target_index)

            # Apply the relative phase correction to the control index
            # to account for the global phase shift created by the target
            # gate
            self.Phase(power * np.pi * (global_shift + 0.5), control_index)

    def CYPow(
            self,
            power: float,
            global_shift: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Y^power gate to the circuit.

        Notes
        -----
        The CYPow gate is defined exactly as Google Cirq's controlled `YPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CYPow(power=0.5, control_index=0, target_index=1)
        >>> circuit.CYPow(power=0.5, global_shift=0.5, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CYPow.__name__, params=locals())

        with self.decompose_last(gate):
            self.CRY(power * np.pi, control_index, target_index)

            # Apply the relative phase correction to the control index
            # to account for the global phase shift created by the target
            # gate
            self.Phase(power * np.pi * (global_shift + 0.5), control_index)

    def CZPow(
            self,
            power: float,
            global_shift: float,
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled Z^power gate to the circuit.

        Notes
        -----
        The CZPow gate is defined exactly as Google Cirq's controlled `ZPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CZPow(power=0.5, control_index=0, target_index=1)
        >>> circuit.CZPow(power=0.5, global_shift=0.5, control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CZPow.__name__, params=locals())

        with self.decompose_last(gate):
            self.CRZ(power * np.pi, control_index, target_index)

            # Apply the relative phase correction to the control index
            # to account for the global phase shift created by the target
            # gate
            self.Phase(power * np.pi * (global_shift + 0.5), control_index)

    def CRXX(
            self,
            angle: float,
            control_index: int,
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Controlled RXX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CRXX(angle=np.pi/2, control_index=0,
        ...              first_target_index=1, second_target_index=2)
        """
        self.MCRXX(
            angle=angle,
            control_indices=control_index,
            first_target_index=first_target_index,
            second_target_index=second_target_index
        )

    def CRYY(
            self,
            angle: float,
            control_index: int,
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Controlled RYY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CRYY(angle=np.pi/2, control_index=0,
        ...              first_target_index=1, second_target_index=2)
        """
        self.MCRYY(
            angle=angle,
            control_indices=control_index,
            first_target_index=first_target_index,
            second_target_index=second_target_index
        )

    def CRZZ(
            self,
            angle: float,
            control_index: int,
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Controlled RZZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_index` : int
            The index of the control qubit.
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CRZZ(angle=np.pi/2, control_index=0,
        ...              first_target_index=1, second_target_index=2)
        """
        self.MCRZZ(
            angle=angle,
            control_indices=control_index,
            first_target_index=first_target_index,
            second_target_index=second_target_index
        )

    def CU3(
            self,
            angles: Sequence[float],
            control_index: int,
            target_index: int
        ) -> None:
        """ Apply a Controlled U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `control_index` : int
            The index of the control qubit.
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_index=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.CU3.__name__, params=locals())

        with self.decompose_last(gate):
            # Explicit optimal implementation covered in MCU3
            self.MCU3(angles, control_index, target_index)

    def CSWAP(
            self,
            control_index: int,
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Controlled SWAP gate to the circuit.

        Parameters
        ----------
        `control_index` : int
            The index of the control qubit.
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.CSWAP(control_index=0, first_target_index=1, second_target_index=2)
        """
        self.MCSWAP(
            control_indices=control_index,
            first_target_index=first_target_index,
            second_target_index=second_target_index
        )

    def MCX(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Pauli-X gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCX(control_indices=0, target_indices=1)
        >>> circuit.MCX(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCX(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCX(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCX.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.H(target_indices)
            self.MCPhase(np.pi, control_indices, target_indices)
            self.H(target_indices)

    def MCY(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Pauli-Y gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCY(control_indices=0, target_indices=1)
        >>> circuit.MCY(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCY(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCY(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCY.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.Sdg(target_indices)
            self.MCX(control_indices, target_indices)
            self.S(target_indices)

    def MCZ(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Pauli-Z gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCZ(control_indices=0, target_indices=1)
        >>> circuit.MCZ(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCZ(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCZ(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCZ.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.H(target_indices)
            self.MCX(control_indices, target_indices)
            self.H(target_indices)

    def MCH(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Hadamard gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCH(control_indices=0, target_indices=1)
        >>> circuit.MCH(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCH(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCH(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCH.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.S(target_indices)
            self.H(target_indices)
            self.T(target_indices)
            self.MCX(control_indices, target_indices)
            self.Tdg(target_indices)
            self.H(target_indices)
            self.Sdg(target_indices)

    def MCS(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Clifford-S gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCS(control_indices=0, target_indices=1)
        >>> circuit.MCS(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCS(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCS(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCS.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.MCPhase(np.pi/2, control_indices, target_indices)

    def MCSdg(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Clifford-S^{dagger} gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCSdg(control_indices=0, target_indices=1)
        >>> circuit.MCSdg(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCSdg(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCSdg(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCSdg.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.MCPhase(-np.pi/2, control_indices, target_indices)

    def MCT(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Clifford-T gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCT(control_indices=0, target_indices=1)
        >>> circuit.MCT(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCT(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCT(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCT.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.MCPhase(np.pi/4, control_indices, target_indices)

    def MCTdg(
            self,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Clifford-T^{dagger} gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCTdg(control_indices=0, target_indices=1)
        >>> circuit.MCTdg(control_indices=0, target_indices=[1, 2])
        >>> circuit.MCTdg(control_indices=[0, 1], target_indices=2)
        >>> circuit.MCTdg(control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCTdg.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.MCPhase(-np.pi/4, control_indices, target_indices)

    def MCRX(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled RX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCRX(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCRX(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCRX(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCRX(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCRX.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            for target_index in target_indices:
                MCRX(self, angle, list(control_indices), target_index)

    def MCRY(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled RY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCRY(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCRY(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCRY(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCRY(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCRY.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            for target_index in target_indices:
                MCRY(self, angle, list(control_indices), target_index)

    def MCRZ(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled RZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCRZ(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCRZ.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            for target_index in target_indices:
                MCRZ(self, angle, list(control_indices), target_index)

    def MCPhase(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Phase gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCPhase(angle=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit.MCPhase(angle=np.pi/2, control_indices=0, target_indices=[1, 2])
        >>> circuit.MCPhase(angle=np.pi/2, control_indices=[0, 1], target_indices=2)
        >>> circuit.MCPhase(angle=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCPhase.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        num_controls = len(control_indices)

        with self.decompose_last(gate):
            for target_index in target_indices:
                new_target_index = target_index
                new_control_indices = list(control_indices).copy()

                for k in range(num_controls):
                    self.MCRZ(angle / (2**k), new_control_indices, new_target_index)
                    new_target_index = new_control_indices.pop()

                self.Phase(angle / 2**num_controls, new_target_index)

    def MCXPow(
            self,
            power: float,
            global_shift: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled X^power gate to the circuit.

        Notes
        -----
        The MCXPow gate is defined exactly as Google Cirq's controlled `XPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCXPow(power=0.5, control_indices=0, target_indices=1)
        >>> circuit.MCXPow(power=0.5, global_shift=0.5, control_indices=0, target_indices=1)
        """
        gate = self.process_gate_params(gate=self.MCXPow.__name__, params=locals())

        with self.decompose_last(gate):
            control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
            target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

            for target_index in target_indices:
                self.MCRX(power * np.pi, control_indices=control_indices, target_indices=target_index)

                # Apply the relative phase correction to the control indices
                # to account for the global phase shift created by the target
                # gate
                if len(control_indices) > 1:
                    self.MCPhase(
                        power * np.pi * (global_shift + 0.5),
                        control_indices=control_indices[:-1],
                        target_indices=control_indices[-1]
                    )
                else:
                    self.Phase(power * np.pi * (global_shift + 0.5), control_indices[0])

    def MCYPow(
            self,
            power: float,
            global_shift: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Y^power gate to the circuit.

        Notes
        -----
        The MCYPow gate is defined exactly as Google Cirq's controlled `YPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCYPow(power=0.5, control_indices=0, target_indices=1)
        >>> circuit.MCYPow(power=0.5, global_shift=0.5, control_indices=0, target_indices=1)
        """
        gate = self.process_gate_params(gate=self.MCYPow.__name__, params=locals())

        with self.decompose_last(gate):
            control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
            target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

            for target_index in target_indices:
                self.MCRY(power * np.pi, control_indices=control_indices, target_indices=target_index)

                # Apply the relative phase correction to the control indices
                # to account for the global phase shift created by the target
                # gate
                if len(control_indices) > 1:
                    self.MCPhase(
                        power * np.pi * (global_shift + 0.5),
                        control_indices=control_indices[:-1],
                        target_indices=control_indices[-1]
                    )
                else:
                    self.Phase(power * np.pi * (global_shift + 0.5), control_indices[0])

    def MCZPow(
            self,
            power: float,
            global_shift: float,
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled Z^power gate to the circuit.

        Notes
        -----
        The MCZPow gate is defined exactly as Google Cirq's controlled `ZPowGate`.

        Parameters
        ----------
        `power` : float
            The power of the gate.
        `global_shift` : float
            The global phase shift of the gate.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Power must be a float or integer.
            - Global shift must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCZPow(power=0.5, control_indices=0, target_indices=1)
        >>> circuit.MCZPow(power=0.5, global_shift=0.5, control_indices=0, target_indices=1)
        """
        gate = self.process_gate_params(gate=self.MCZPow.__name__, params=locals())

        with self.decompose_last(gate):
            control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
            target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

            for target_index in target_indices:
                self.MCRZ(power * np.pi, control_indices=control_indices, target_indices=target_index)

                # Apply the relative phase correction to the control indices
                # to account for the global phase shift created by the target
                # gate
                if len(control_indices) > 1:
                    self.MCPhase(
                        power * np.pi * (global_shift + 0.5),
                        control_indices=control_indices[:-1],
                        target_indices=control_indices[-1]
                    )
                else:
                    self.Phase(power * np.pi * (global_shift + 0.5), control_indices[0])

    def MCRXX(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Multi-Controlled RXX gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCRXX(angle=np.pi/2, control_indices=0,
        ...               first_target_index=1, second_target_index=2)
        >>> circuit.MCRXX(angle=np.pi/2, control_indices=[0, 1],
        ...               first_target_index=2, second_target_index=3)
        """
        gate = self.process_gate_params(gate=self.MCRXX.__name__, params=locals())

        with self.decompose_last(gate):
            control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

            self.MCH(control_indices=control_indices, target_indices=first_target_index)
            self.MCH(control_indices=control_indices, target_indices=second_target_index)
            self.MCX(
                control_indices=list(control_indices) + [first_target_index],
                target_indices=second_target_index
            )
            self.MCRZ(angle, control_indices=control_indices, target_indices=second_target_index)
            self.MCX(
                control_indices=list(control_indices) + [first_target_index],
                target_indices=second_target_index
            )
            self.MCH(control_indices=control_indices, target_indices=first_target_index)
            self.MCH(control_indices=control_indices, target_indices=second_target_index)

    def MCRYY(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Multi-Controlled RYY gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCRYY(angle=np.pi/2, control_indices=0,
        ...               first_target_index=1, second_target_index=2)
        >>> circuit.MCRYY(angle=np.pi/2, control_indices=[0, 1],
        ...               first_target_index=2, second_target_index=3)
        """
        gate = self.process_gate_params(gate=self.MCRYY.__name__, params=locals())

        with self.decompose_last(gate):
            control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

            self.MCRX(
                angle=np.pi/2,
                control_indices=control_indices,
                target_indices=first_target_index
            )
            self.MCRX(
                angle=np.pi/2,
                control_indices=control_indices,
                target_indices=second_target_index
            )
            self.MCX(
                control_indices=list(control_indices) + [first_target_index],
                target_indices=second_target_index
            )
            self.MCRZ(angle, control_indices=control_indices, target_indices=second_target_index)
            self.MCX(
                control_indices=list(control_indices) + [first_target_index],
                target_indices=second_target_index
            )
            self.MCRX(
                angle=-np.pi/2,
                control_indices=control_indices,
                target_indices=first_target_index
            )
            self.MCRX(
                angle=-np.pi/2,
                control_indices=control_indices,
                target_indices=second_target_index
            )

    def MCRZZ(
            self,
            angle: float,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Multi-Controlled RZZ gate to the circuit.

        Parameters
        ----------
        `angle` : float
            The rotation angle in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCRZZ(angle=np.pi/2, control_indices=0,
        ...               first_target_index=1, second_target_index=2)
        >>> circuit.MCRZZ(angle=np.pi/2, control_indices=[0, 1],
        ...               first_target_index=2, second_target_index=3)
        """
        gate = self.process_gate_params(gate=self.MCRZZ.__name__, params=locals())

        with self.decompose_last(gate):
            control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

            self.MCX(
                control_indices=list(control_indices) + [first_target_index],
                target_indices=second_target_index
            )
            self.MCRZ(angle, control_indices=control_indices, target_indices=second_target_index)
            self.MCX(
                control_indices=list(control_indices) + [first_target_index],
                target_indices=second_target_index
            )

    def MCU3(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:
        """ Apply a Multi-Controlled U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_indices` : int | Sequence[int]
            The index of the target qubit(s).

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=0, target_indices=1)
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=0, target_indices=[1, 2])
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=[0, 1], target_indices=2)
        >>> circuit.MCU3(angles=[np.pi/2, np.pi/2, np.pi/2], control_indices=[0, 1], target_indices=[2, 3])
        """
        gate = self.process_gate_params(gate=self.MCU3.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        with self.decompose_last(gate):
            self.MCPhase(angles[2], control_indices, target_indices)
            self.MCRY(angles[0], control_indices, target_indices)
            self.MCPhase(angles[1], control_indices, target_indices)

    def MCSWAP(
            self,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:
        """ Apply a Controlled SWAP gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `first_target_index` : int
            The index of the first target qubit.
        `second_target_index` : int
            The index of the second target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.MCSWAP(control_indices=0, first_target_index=1, second_target_index=2)
        >>> circuit.MCSWAP(control_indices=[0, 1], first_target_index=2, second_target_index=3)
        """
        gate = self.process_gate_params(gate=self.MCSWAP.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else list(control_indices)

        with self.decompose_last(gate):
            self.CX(second_target_index, first_target_index)
            self.MCX(control_indices + [first_target_index], second_target_index)
            self.CX(second_target_index, first_target_index)

    def UCPauliRot(
            self,
            angles: Sequence[float],
            rot_axis: Literal["X", "Y", "Z"],
            control_indices: int | Sequence[int],
            target_index: int
        ) -> None:
        """ Apply a uniformly controlled Pauli rotation to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `rot_axis` : Literal["X", "Y", "Z"]
            The rotation axis.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.
            - The number of angles must be a power of 2.
            - The number of control qubits must be equal to the number of angles.
            - Invalid rotation axis. Expected 'X', 'Y' or 'Z'.

        Usage
        -----
        >>> circuit.UCPauliRot(control_indices=0, target_index=1,
        ...                    angles=[np.pi/2, np.pi/2], rot_axis="X")
        """
        if isinstance(control_indices, int):
            control_indices = [control_indices]

        angles_params = np.asarray(angles)

        # Check if the number of angles is a power of 2
        is_power_of_2 = (angles_params.shape[0] & (angles_params.shape[0]-1) == 0) \
            and angles_params.shape[0] != 0

        if not is_power_of_2:
            raise ValueError(
                f"The number of angles must be a power of 2. Received {len(angles_params)}.")

        num_control_qubits = len(control_indices)

        # Check if the number of control qubits is equal to the number of angles
        if num_control_qubits != np.log2(len(angles_params)):
            raise ValueError(
                f"The number of control qubits must be equal to the number of angles. "
                f"Received {num_control_qubits} control qubits and {len(angles_params)} angles.")

        # Check if the rotation axis is valid
        if rot_axis not in ["X", "Y", "Z"]:
            raise ValueError(
                f"Invalid rotation axis. Expected 'X', 'Y' or 'Z'. Received {rot_axis}.")

        gate_mapping = {
            "X": lambda: self.RX,
            "Y": lambda: self.RY,
            "Z": lambda: self.RZ
        }

        # If there are no control qubits, apply the gate directly
        # to the target qubit with the first angle
        if num_control_qubits == 0:
            gate_mapping[rot_axis]()(angles_params[0], target_index)
            return

        # Make a copy of the angles parameters to avoid modifying the original
        angles_params = angles_params.copy()

        decompose_uc_rotations(angles_params, 0, len(angles_params), False)

        # Apply the uniformly controlled Pauli rotations
        for i, angle in enumerate(angles_params):
            gate_mapping[rot_axis]()(angle, target_index)

            # Determine the index of the qubit we want to control the CX gate
            # Note that it corresponds to the number of trailing zeros in the
            # binary representation of i+1
            if not i == len(angles_params) - 1:
                binary_rep = np.binary_repr(i + 1)
                control_index = len(binary_rep) - len(binary_rep.rstrip("0"))
            else:
                control_index = num_control_qubits - 1

            # For X rotations, we have to additionally place some RY gates around the
            # CX gates
            # They change the basis of the NOT operation, such that the
            # decomposition of for uniformly controlled X rotations works correctly by symmetry
            # with the decomposition of uniformly controlled Z or Y rotations
            if rot_axis == "X":
                self.RY(np.pi / 2, target_index)
            self.CX(control_indices[control_index], target_index)
            if rot_axis == "X":
                self.RY(-np.pi / 2, target_index)

    def UCRX(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_index: int
        ) -> None:
        """ Apply a uniformly controlled RX gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.
            - The number of angles must be a power of 2.
            - The number of control qubits must be equal to the number of angles.

        Usage
        -----
        >>> circuit.UCRX(angles=[np.pi/2, np.pi/2], control_indices=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.UCRX.__name__, params=locals())

        with self.decompose_last(gate):
            self.UCPauliRot(angles, "X", control_indices, target_index)

    def UCRY(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_index: int
        ) -> None:
        """ Apply a uniformly controlled RY gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.
            - The number of angles must be a power of 2.
            - The number of control qubits must be equal to the number of angles.

        Usage
        -----
        >>> circuit.UCRY(angles=[np.pi/2, np.pi/2], control_indices=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.UCRY.__name__, params=locals())

        with self.decompose_last(gate):
            self.UCPauliRot(angles, "Y", control_indices, target_index)

    def UCRZ(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_index: int
        ) -> None:
        """ Apply a uniformly controlled RZ gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.
            - The number of angles must be a power of 2.
            - The number of control qubits must be equal to the number of angles.

        Usage
        -----
        >>> circuit.UCRZ(angles=[np.pi/2, np.pi/2], control_indices=0, target_index=1)
        """
        gate = self.process_gate_params(gate=self.UCRZ.__name__, params=locals())

        with self.decompose_last(gate):
            self.UCPauliRot(angles, "Z", control_indices, target_index)

    def Diagonal(
            self,
            diagnoal: NDArray[np.complex128],
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Apply a diagonal gate to the circuit.

        Notes
        -----
        .. math::
            \text{DiagonalGate}\ q_0, q_1, .., q_{n-1} =
                \begin{pmatrix}
                    D[0]    & 0         & \dots     & 0 \\
                    0       & D[1]      & \dots     & 0 \\
                    \vdots  & \vdots    & \ddots    & 0 \\
                    0       & 0         & \dots     & D[n-1]
                \end{pmatrix}

        Diagonal gates are useful as representations of Boolean functions,
        as they can map from :math:`\{0,1\}^{2^n}` to :math:`\{0,1\}^{2^n}` space. For example a phase
        oracle can be seen as a diagonal gate with :math:`\{1, -1\}` on the diagonals. Such
        an oracle will induce a :math:`+1` or :math`-1` phase on the amplitude of any corresponding
        basis state.

        Diagonal gates appear in many classically hard oracular problems such as
        Forrelation or Hidden Shift circuits.

        Diagonal gates are represented and simulated more efficiently than a dense
        :math:`2^n \times 2^n` unitary matrix.

        The reference implementation is via the method described in
        Theorem 7 of [1]. The code is based on Emanuel Malvetti's semester thesis
        at ETH in 2018, supervised by Raban Iten and Prof. Renato Renner.

        [1] Shende, Bullock, Markov,
        Synthesis of Quantum Logic Circuits (2009).
        https://arxiv.org/pdf/quant-ph/0406176.pdf

        Parameters
        ----------
        `diagnoal` : NDArray[np.complex128]
            The diagonal matrix to apply to the circuit.

        Raises
        ------
        ValueError
            - The number of diagonal entries is not a positive power of 2.
            - The number of qubits passed must be the same as the number of qubits needed to prepare the diagonal.
            - A diagonal element does not have absolute value one.

        Usage
        -----
        >>> circuit.Diagnoal([[1, 0],
        ...                   [0, 1]])
        """
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Check if the number of diagonal entries is a power of 2
        num_qubits = np.log2(len(diagnoal))

        if num_qubits < 1 or not int(num_qubits) == num_qubits:
            raise ValueError("The number of diagonal entries is not a positive power of 2.")
        num_qubits = int(num_qubits)

        if num_qubits != len(qubit_indices):
            raise ValueError(
                "The number of qubits passed must be the same as "
                "the number of qubits needed to prepare the diagonal."
            )

        if not np.allclose(np.abs(diagnoal), 1, atol=1e-10):
            raise ValueError("A diagonal element does not have absolute value one.")

        # Since the diagonal is a unitary, all its entries have absolute value
        # one and the diagonal is fully specified by the phases of its entries.
        diagonal_phases = [cmath.phase(z) for z in diagnoal]
        n = len(diagnoal)

        while n >= 2:
            angles_rz = []

            for i in range(0, n, 2):
                diagonal_phases[i // 2], rz_angle = extract_rz(
                    diagonal_phases[i], diagonal_phases[i + 1]
                )
                angles_rz.append(rz_angle)

            num_act_qubits = int(np.log2(n))
            control_indices = list(range(num_qubits - num_act_qubits + 1, num_qubits))
            target_index = num_qubits - num_act_qubits

            control_indices = [qubit_indices[i] for i in control_indices]
            target_index = qubit_indices[target_index]

            self.UCRZ(
                angles=angles_rz,
                control_indices=control_indices,
                target_index=target_index
            )

            n //= 2

        self.GlobalPhase(diagonal_phases[0])

    def UC(
            self,
            gates: list[NDArray[np.complex128]],
            control_indices: int | Sequence[int],
            target_index: int,
            up_to_diagonal: bool=False,
            multiplexor_simplification: bool=True
        ) -> None:
        """ Apply a uniformly controlled gate (multiplexor) to the circuit.

        Notes
        -----
        The decomposition used in this method is based on the paper by Bergholm et al. [1].
        Additional simplifications were made by de Carvalho et al. [2].

        [1] Bergholm,
        Quantum circuits with uniformly controlled one-qubit gates (2005).
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330

        [2] de Carvalho, Batista, de Veras, Araujo, da Silva,
        Quantum multiplexer simplification for state preparation (2024).
        https://arxiv.org/abs/2409.05618

        Parameters
        ----------
        `gates` : list[NDArray[np.complex128]]
            The gates to apply to the circuit.
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.
        `up_to_diagonal` : bool, optional, default=False
            Determines if the gate is implemented up to a diagonal
            or if it is decomposed completely.
        `multiplexor_simplification` : bool, optional, default=True
            Determines if the multiplexor is simplified using [2].

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.
            - The number of single-qubit gates must be a non-negative power of 2.
            - The number of control qubits passed must be equal to the number of gates.
            - A gate is not unitary.

        Usage
        -----
        >>> circuit.UC(control_indices=[1, 2], target_index=0,
        ...            [[[1, 0],
        ...              [0, 1]],
        ...             [[0, 1],
        ...              [1, 0]]])
        >>> circuit.UC(control_indices=[1, 2], target_index=0,
        ...            [[[1, 0],
        ...              [0, 1]],
        ...             [[0, 1],
        ...              [1, 0]]], up_to_diagonal=True)
        """
        if isinstance(control_indices, int):
            control_indices = [control_indices]

        for gate in gates:
            if not gate.shape == (2, 2):
                raise ValueError(f"The dimension of a gate is not equal to 2x2. Received {gate.shape}.")

        # Check if number of gates in gate_list is a positive power of two
        num_control = np.log2(len(gates))
        if num_control < 0 or not int(num_control) == num_control:
            raise ValueError(
                "The number of single-qubit gates is not a non-negative power of 2."
            )

        if not num_control == len(control_indices):
            raise ValueError(
                "The number of control qubits passed must be equal to the number of gates."
            )

        # Check if the single-qubit gates are unitaries
        for gate in gates:
            if not is_unitary_matrix(gate, 1e-10):
                raise ValueError("A gate is not unitary.")

        qubits = [target_index] + list(control_indices)
        if multiplexor_simplification:
            new_controls, gates = simplify(gates, int(num_control))
            control_indices = [qubits[len(control_indices) + 1 - i] for i in new_controls]
            control_indices.reverse()

        # If there is no control, we use the ZYZ decomposition
        if not control_indices:
            self.unitary(gates[0], target_index)
            return

        # If there is at least one control, first,
        # we find the single qubit gates of the decomposition
        (single_qubit_gates, diagonal) = decompose_ucg_help(gates, len(control_indices) + 1)

        # Now, it is easy to place the CX gates and some Hadamards and RZ(pi/2) gates
        # (which are absorbed into the single-qubit unitaries) to get back the full decomposition.
        for i, gate in enumerate(single_qubit_gates):
            if i == 0:
                self.unitary(gate, target_index)
                self.H(target_index)

            elif i == len(single_qubit_gates) - 1:
                self.H(target_index)
                self.RZ(-np.pi / 2, target_index)
                self.unitary(gate, target_index)

            else:
                self.H(target_index)
                self.RZ(-np.pi / 2, target_index)
                self.unitary(gate, target_index)
                self.H(target_index)

            # The number of the control qubit is given by the number of zeros at the end
            # of the binary representation of (i+1)
            binary_rep = np.binary_repr(i + 1)
            num_trailing_zeros = len(binary_rep) - len(binary_rep.rstrip("0"))
            control_index = num_trailing_zeros

            # Add CX gate
            if not i == len(single_qubit_gates) - 1:
                self.CX(control_indices[control_index], target_index)
                self.GlobalPhase(-np.pi/4)

        # If `up_to_diagonal` is False, we apply the diagonal gate
        if not up_to_diagonal:
            self.Diagonal(diagonal, qubit_indices=[target_index] + list(control_indices))

    @abstractmethod
    def GlobalPhase(
            self,
            angle: float
        ) -> None:
        """ Apply a global phase to the circuit.

        Parameters
        ----------
        `angle` : float
            The global phase to apply to the circuit.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Angle must be a float or integer.
        ValueError
            - Qubit index out of range.

        Usage
        -----
        >>> circuit.GlobalPhase(angle=np.pi/2)
        """

    def QFT(
            self,
            qubit_indices: int | Sequence[int],
            do_swaps: bool=True,
            approximation_degree: int=0,
            inverse: bool=False
        ) -> None:
        """ Apply the Quantum Fourier Transform to the circuit.

        Notes
        -----
        The Quantum Fourier Transform (QFT) is a linear transformation that maps
        a quantum state to its frequency domain representation. It is the quantum
        analogue of the Discrete Fourier Transform (DFT). The QFT is an important
        subroutine in many quantum algorithms, including Shor's algorithm for
        integer factorization and quantum phase estimation.

        The QFT is defined as:

        .. math::
            |x\\rangle \\rightarrow \\frac{1}{\\sqrt{N}} \\sum_{y=0}^{N-1} e^{2\\pi ixy/N} |y\\rangle

        The QFT can be implemented using a series of Hadamard gates and controlled
        phase rotations. The QFT can be efficiently implemented on a quantum computer
        using a recursive decomposition method.

        For more information on the approximate decomposition of QFT, refer to:
        https://arxiv.org/pdf/quant-ph/0403071

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.
        `do_swaps` : bool, optional, default=True
            Whether to apply the SWAP gates at the end of the QFT.
        `approximation_degree` : int, optional, default=0
            The degree of approximation to use in the QFT.
        `inverse` : bool, optional, default=False
            Whether to apply the inverse QFT.

        Raises
        ------
        TypeError
            - Approximation degree must be an integer.
        ValueError
            - Qubit index out of range.
            - Approximation degree must be non-negative.
        """
        if not isinstance(approximation_degree, int):
            raise TypeError("Approximation degree must be an integer.")

        if approximation_degree < 0:
            raise ValueError("Approximation degree must be non-negative.")

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        num_qubits = len(qubit_indices)
        circuit: Circuit = type(self)(num_qubits)

        for j in reversed(range(num_qubits)):
            circuit.H(j)
            num_entanglements = max(0, j - max(0, approximation_degree - (num_qubits - j - 1)))
            for k in reversed(range(j - num_entanglements, j)):
                # Use negative exponents so that the angle safely underflows to zero, rather than
                # using a temporary variable that overflows to infinity in the worst case
                lam = np.pi * (2.0 ** (k - j))
                circuit.CPhase(lam, j, k)

        if do_swaps:
            for i in range(num_qubits // 2):
                circuit.SWAP(i, num_qubits - i - 1)

        if inverse:
            circuit.horizontal_reverse()

        self.add(circuit, qubit_indices)

    def initialize(
            self,
            state: NDArray[np.complex128] | Bra | Ket,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Initialize the state of the circuit.

        Parameters
        ----------
        `state` : NDArray[np.complex128] | qickit.primitives.Bra | qickit.primitives.Ket
            The state to initialize the circuit to.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - If the state is not a numpy array or a Bra/Ket object.
            - If the qubit indices are not integers or a sequence of integers.
        ValueError
            - If the compression percentage is not in the range [0, 100].
            - If the index type is not "row" or "snake".
            - If the number of qubit indices is not equal to the number of qubits in the state.
        IndexError
            - If the qubit indices are out of range.

        Usage
        -----
        >>> circuit.initialize([1, 0], qubit_indices=0)
        """
        # Initialize the state preparation schema
        isometry = Isometry(output_framework=type(self))

        # Prepare the state
        self = isometry.apply_state(
            circuit=self,
            state=state,
            qubit_indices=qubit_indices
        )

    def unitary(
            self,
            unitary_matrix: NDArray[np.complex128] | Operator,
            qubit_indices:  int | Sequence[int]
        ) -> None:
        """ Apply a unitary gate to the circuit. Uses Quantum Shannon Decomposition
        to decompose the unitary matrix into RY, RZ, and CX gates.

        Parameters
        ----------
        `unitary_matrix` : NDArray[np.complex128] | qickit.primitives.Operator
            The unitary matrix to apply to the circuit.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        TypeError
            - If the unitary is not a numpy array or an Operator object.
            - If the qubit indices are not integers or a sequence of integers.
        ValueError
            - If the number of qubit indices is not equal to the number of qubits
            in the unitary operator.
        IndexError
            - If the qubit indices are out of range.

        Usage
        -----
        >>> circuit.unitary([[0, 1],
        ...                  [1, 0]], qubit_indices=0)
        >>> circuit.unitary([[0, 0, 0, 1],
        ...                  [0, 0, 1, 0],
        ...                  [0, 1, 0, 0],
        ...                  [1, 0, 0, 0]], qubit_indices=[0, 1])
        """
        # Initialize the unitary preparation schema
        unitary_preparer = ShannonDecomposition(output_framework=type(self))

        # Prepare the unitary matrix
        self = unitary_preparer.apply_unitary(
            circuit=self,
            unitary=unitary_matrix,
            qubit_indices=qubit_indices
        )

    def vertical_reverse(self) -> None:
        """ Perform a vertical reverse operation.

        Usage
        -----
        >>> circuit.vertical_reverse()
        """
        self.change_mapping(list(range(self.num_qubits))[::-1])

    def horizontal_reverse(
            self,
            adjoint: bool=True
        ) -> None:
        """ Perform a horizontal reverse operation. This is equivalent
        to the adjoint of the circuit if `adjoint=True`. Otherwise, it
        simply reverses the order of the operations.

        Parameters
        ----------
        `adjoint` : bool
            Whether or not to apply the adjoint of the circuit.

        Raises
        ------
        TypeError
            - Adjoint must be a boolean.

        Usage
        -----
        >>> circuit.horizontal_reverse()
        >>> circuit.horizontal_reverse(adjoint=False)
        """
        if not isinstance(adjoint, bool):
            raise TypeError("Adjoint must be a boolean.")

        # Reverse the order of the operations
        self.circuit_log = self.circuit_log[::-1]

        # If adjoint is True, then multiply the angles by -1
        if adjoint:
            # Iterate over every operation, and change the index accordingly
            for operation in self.circuit_log:
                if "angle" in operation:
                    operation["angle"] = -operation["angle"]
                elif "angles" in operation:
                    operation["angles"] = [-operation["angles"][0], -operation["angles"][2], -operation["angles"][1]]
                elif "power" in operation:
                    operation["power"] = -operation["power"]
                elif operation["gate"] in ["Sdg", "Tdg", "CSdg", "CTdg", "MCSdg", "MCTdg"]:
                    operation["gate"] = operation["gate"].replace("dg", "")
                elif operation["gate"] in ["S", "T", "CS", "CT", "MCS", "MCT"]:
                    operation["gate"] = operation["gate"] + "dg"

        self.update()

    def add(
            self,
            circuit: Circuit,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Append two circuits together in a sequence.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to append to the current circuit.
        `qubit_indices` : int | Sequence[int]
            The indices of the qubits to add the circuit to.

        Raises
        ------
        TypeError
            - The circuit must be a Circuit object.
            - Qubit index must be an integer.
        ValueError
            - The number of qubits must match the number of qubits in the `circuit`.

        Usage
        -----
        >>> circuit.add(circuit=circuit2, qubit_indices=0)
        >>> circuit.add(circuit=circuit2, qubit_indices=[0, 1])
        """
        if not isinstance(circuit, Circuit):
            raise TypeError("The circuit must be a Circuit object.")

        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        if isinstance(qubit_indices, Sequence):
            if not all(isinstance(qubit_index, int) for qubit_index in qubit_indices):
                raise TypeError("Qubit index must be an integer.")
            if len(qubit_indices) != circuit.num_qubits:
                raise ValueError("The number of qubits must match the number of qubits in the circuit.")

        # Create a copy of the as the `add` method is applied in-place
        circuit_log = copy.deepcopy(circuit.circuit_log)

        for operation in circuit_log:
            for key in set(operation.keys()).intersection(ALL_QUBIT_KEYS):
                if isinstance(operation[key], Sequence):
                    operation[key] = [qubit_indices[index] for index in operation[key]] # type: ignore
                else:
                    operation[key] = list(qubit_indices)[operation[key]] # type: ignore

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in circuit_log:
            # Extract gate name and remove it from gate_info for kwargs
            gate_name = gate_info.pop("gate", None)

            # Extract gate definition and remove it from gate_info for kwargs
            gate_definition = gate_info.pop("definition", None)

            # Use the gate mapping to apply the corresponding gate with remaining kwargs
            getattr(self, gate_name)(**gate_info)

            # Re-insert gate name and definition into gate_info if needed elsewhere
            gate_info["gate"] = gate_name
            gate_info["definition"] = gate_definition

    @abstractmethod
    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Measure the qubits in the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The indices of the qubits to measure.

        Usage
        -----
        >>> circuit.measure(qubit_indices=0)
        >>> circuit.measure(qubit_indices=[0, 1])
        """

    def measure_all(self) -> None:
        """ Measure all the qubits in the circuit.

        Usage
        -----
        >>> circuit.measure_all()
        """
        self.measure(qubit_indices=list(range(self.num_qubits)))

    @abstractmethod
    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `backend` : qickit.backend.Backend, optional
            The backend to run the circuit on.

        Returns
        -------
        `statevector` : NDArray[np.complex128]
            The statevector of the circuit.

        Usage
        -----
        >>> circuit.get_statevector()
        >>> circuit.get_statevector(backend=backend)
        """

    @abstractmethod
    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:
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

        Raises
        ------
        ValueError
            - The circuit must have at least one qubit that is measured.

        Usage
        -----
        >>> circuit.get_counts(num_shots=1024)
        >>> circuit.get_counts(num_shots=1024, backend=backend)
        """

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
        circuit = self.copy()

        # Transpile the circuit to both simplify and optimize it
        circuit.transpile()

        # Get the depth of the circuit
        depth = circuit.get_dag().get_depth()

        return depth

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
    def get_unitary(self) -> NDArray[np.complex128]:
        """ Get the unitary matrix of the circuit.

        Returns
        -------
        `unitary` : NDArray[np.complex128]
            The unitary matrix of the circuit.

        Usage
        -----
        >>> circuit.get_unitary()
        """

    def get_instructions(
            self,
            include_measurements: bool=True
        ) -> list[dict]:
        """ Get the instructions of the circuit.

        Parameters
        ----------
        `include_measurements` : bool, optional
            Whether or not to include the measurement instructions.

        Returns
        -------
        `instructions` : list[dict]
            The instructions of the circuit.
        """
        if include_measurements:
            return self.circuit_log

        instructions = []

        # Filter out the measurement instructions
        for operation in self.circuit_log:
            if not operation["gate"] == "measure":
                instructions.append(operation)

        return instructions

    def get_dag(self) -> DAGCircuit:
        """ Get the DAG representation of the circuit.

        Returns
        -------
        `dag` : qickit.dag.DAGCircuit
            The DAG representation of the circuit.

        Usage
        -----
        >>> dag = circuit.get_dag()
        """
        dag = DAGCircuit(self.num_qubits)

        for operation in self.circuit_log:
            dag.add_operation(operation)

        return dag

    def get_global_phase(self) -> float:
        """ Get the global phase of the circuit.

        Returns
        -------
        `global_phase` : float
            The global phase of the circuit.

        Usage
        -----
        >>> circuit.get_global_phase()
        """
        return np.exp(1j * self.global_phase)

    def count_ops(self) -> dict[str, int]:
        """ Count the operations in the circuit.

        Returns
        -------
        `ops` : dict[str, int]
            The count of operations in the circuit.

        Usage
        -----
        >>> circuit.count_ops()
        """
        ops: dict[str, int] = {}

        for operation in self.circuit_log:
            gate = operation["gate"]
            ops[gate] = ops.get(gate, 0) + 1

        return ops

    @abstractmethod
    def reset_qubit(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:
        """ Reset the qubits in the circuit.

        Parameters
        ----------
        `qubit_indices` : int | Sequence[int]
            The indices of the qubits to reset.

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
            - Qubit index must be a sequence of integers.
        ValueError
            - Qubit index out of range.
        """

    def _remove_measurements_inplace(self) -> None:
        """ Remove the measurement instructions from the circuit inplace.

        Usage
        -----
        >>> circuit.remove_measurements_inplace()
        """
        # Filter out the measurement instructions
        instructions = self.get_instructions(include_measurements=False)

        # Create a new circuit without the measurement instructions
        self.circuit_log = instructions

        self.update()

    def _remove_measurements(self) -> Circuit:
        """ Remove the measurement instructions from the circuit
        and return it as a new instance.

        Usage
        -----
        >>> circuit.remove_measurements_inplace()
        """
        # Filter out the measurement instructions
        instructions = self.get_instructions(include_measurements=False)

        # Create a new circuit without the measurement instructions
        circuit = type(self)(self.num_qubits)
        circuit.circuit_log = instructions

        circuit.update()

        return circuit

    @overload
    def remove_measurements(
            self,
            inplace: Literal[False]
        ) -> Circuit:
        """ Overload of `.remove_measurements` method.
        """

    @overload
    def remove_measurements(
            self,
            inplace: Literal[True]
        ) -> None:
        """ Overload of `.remove_measurements` method.
        """

    def remove_measurements(
            self,
            inplace: bool=False
        ) -> Circuit | None:
        """ Remove the measurement instructions from the circuit.

        Parameters
        ----------
        `inplace` : bool, optional, default=False
            Whether or not to remove the measurement instructions in place.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit | None
            The circuit without the measurement instructions. None is returned
            if `inplace` is set to True.

        Usage
        -----
        >>> new_circuit = circuit.remove_measurements()
        """
        if inplace:
            self._remove_measurements_inplace()
            return None

        return self._remove_measurements()

    def decompose(
            self,
            reps: int=1,
            full: bool=False
        ) -> Circuit:
        """ Decompose the gates in the circuit to their implementation gates.

        Parameters
        ----------
        `reps` : int, optional, default=1
            The number of times to decompose the gates.
        `full` : bool, optional, default=False
            Whether or not to fully decompose the gates.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The circuit with decomposed gates.

        Usage
        -----
        >>> new_circuit = circuit.decompose(reps=3)
        """
        # Create a new circuit to store the decomposed gates
        circuit = type(self)(self.num_qubits)

        # Create a copy of the circuit log to use as placeholder for each layer of decomposition
        circuit_log_copy = copy.deepcopy(self.circuit_log)

        # Iterate over the circuit log, and use the `definition` key to define the decomposition
        # Continue until the circuit log is fully decomposed
        if full:
            while True:
                gates = set([operation["gate"] for operation in circuit_log_copy])

                if gates.issubset(set(["U3", "CX", "GlobalPhase", "measure"])):
                    break

                for operation in circuit_log_copy:
                    if operation["definition"] != []:
                        for op in operation["definition"]:
                            circuit.circuit_log.append(op)
                    else:
                        circuit.circuit_log.append(operation)

                circuit_log_copy = circuit.circuit_log
                circuit.circuit_log = []

        # Iterate over the circuit log, and use the `definition` key to define the decomposition
        # Each rep will decompose the circuit one layer further
        else:
            for _ in range(reps):
                for operation in circuit_log_copy:
                    if operation["definition"] != []:
                        for op in operation["definition"]:
                            circuit.circuit_log.append(op)
                    else:
                        circuit.circuit_log.append(operation)
                circuit_log_copy = circuit.circuit_log
                circuit.circuit_log = []

        circuit.circuit_log = circuit_log_copy

        circuit.update()

        return circuit

    def transpile(
            self,
            direct_transpile: bool=True,
            synthesis_method: UnitaryPreparation | None = None
        ) -> None:
        """ Transpile the circuit to U3 and CX gates.

        Parameters
        ----------
        `direct_transpile` : bool, optional
            Whether or not to directly transpile the circuit. When set to True,
            we wil directly pass a `qickit.circuit.QiskitCircuit` object to the
            transpiler, which will directly transpile the circuit to U3 and CX
            gates. This is significantly more efficient as compared to first
            getting the unitary, applying the unitary to the circuit, and then
            synthesizing the unitary.
        `synthesis_method` : qickit.circuit.UnitaryPreparation, optional
            The method to use for synthesizing the unitary. This is only used
            when `direct_transpile` is set to False.

        Usage
        -----
        >>> circuit.transpile()
        """
        from qickit.optimizer import QiskitTranspiler

        if direct_transpile:
            transpiled_circuit = type(self)(self.num_qubits)
            transpiler = QiskitTranspiler()
            transpiled_circuit = transpiler.optimize(self)

        else:
            if synthesis_method is None:
                synthesis_method = QiskitUnitaryTranspiler(output_framework=type(self))

            unitary_matrix = self.get_unitary()
            transpiled_circuit = synthesis_method.prepare_unitary(unitary_matrix)

        # Update the circuit
        self.circuit_log = transpiled_circuit.circuit_log
        self.circuit = transpiled_circuit.circuit

    def compress(
            self,
            compression_percentage: float
        ) -> None:
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
        # compression percentage to 0 (this means the gate does nothing, and can be removed)
        for index, operation in enumerate(self.circuit_log):
            if "angle" in operation:
                if abs(operation["angle"]) < threshold:
                    indices_to_remove.append(index)
            elif "angles" in operation:
                if all([abs(angle) < threshold for angle in operation["angles"]]):
                    indices_to_remove.append(index)

        # Remove the operations with angles within the compression percentage
        for index in sorted(indices_to_remove, reverse=True):
            del self.circuit_log[index]

        self.update()

    def change_mapping(
            self,
            qubit_indices: Sequence[int]
        ) -> None:
        """ Change the mapping of the circuit.

        Parameters
        ----------
        `qubit_indices` : Sequence[int]
            The updated order of the qubits.

        Raises
        ------
        TypeError
            - Qubit indices must be a collection of integers.
        ValueError
            - The number of qubits must match the number of qubits in the circuit.

        Usage
        -----
        >>> circuit.change_mapping(qubit_indices=[1, 0])
        """
        if not all(isinstance(index, int) for index in qubit_indices):
            raise TypeError("Qubit indices must be a collection of integers.")

        if isinstance(qubit_indices, Sequence):
            qubit_indices = list(qubit_indices)
        elif isinstance(qubit_indices, np.ndarray):
            qubit_indices = qubit_indices.tolist()

        if self.num_qubits != len(qubit_indices):
            raise ValueError("The number of qubits must match the number of qubits in the circuit.")

        # Update the qubit indices
        for operation in self.circuit_log:
            for key in set(operation.keys()).intersection(ALL_QUBIT_KEYS):
                if isinstance(operation[key], list):
                    operation[key] = [qubit_indices[index] for index in operation[key]]
                else:
                    operation[key] = qubit_indices[operation[key]]

        self.update()

    def convert(
            self,
            circuit_framework: Type[Circuit]
        ) -> Circuit:
        """ Convert the circuit to another circuit framework.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The circuit framework to convert to.

        Returns
        -------
        `converted_circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circle.Circuit`.

        Usage
        -----
        >>> converted_circuit = circuit.convert(circuit_framework=QiskitCircuit)
        """
        if not issubclass(circuit_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        # Define the new circuit using the provided framework
        converted_circuit = circuit_framework(self.num_qubits)

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in self.circuit_log:
            # Extract gate name and remove it from gate_info for kwargs
            gate_name = gate_info.pop("gate", None)

            # Extract gate definition and remove it from gate_info for kwargs
            gate_definition = gate_info.pop("definition", None)

            # Use the gate mapping to apply the corresponding gate with remaining kwargs
            getattr(converted_circuit, gate_name)(**gate_info)

            # Re-insert gate name and definition into gate_info if needed elsewhere
            gate_info["gate"] = gate_name
            gate_info["definition"] = gate_definition

        return converted_circuit

    def control(
            self,
            num_controls: int
        ) -> Circuit:
        """ Make the circuit into a controlled operation.

        Note
        ----
        This method is used to create a controlled version of the circuit.
        This can be understood as converting single qubit gates to controlled
        (or multi-controlled) gates, and controlled gates to multi-controlled
        gates.

        Parameters
        ----------
        `num_controls` : int
            The number of control qubits.

        Returns
        -------
        `controlled_circuit` : qickit.circuit.Circuit
            The circuit as a controlled gate.
        """
        # Create a copy of the circuit
        circuit = self.copy()

        # When a target gate has global phase, we need to account for that by resetting
        # the global phase, and then applying it to the control indices using the Phase
        # or MCPhase gates depending on the number of control indices
        circuit.circuit_log = [op for op in circuit.circuit_log if op["gate"] != "GlobalPhase"]

        # Define a controlled circuit
        controlled_circuit = type(circuit)(num_qubits=circuit.num_qubits + num_controls)

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in circuit.circuit_log:
            # Extract gate name and remove it from gate_info for kwargs
            gate_name = gate_info.pop("gate")

            # Extract gate definition and remove it from gate_info for kwargs
            gate_definition = gate_info.pop("definition", None)

            # Change the gate name from single qubit and controlled to multi-controlled
            gate_name = f"{dict(C='M', M='').get(gate_name[0], 'MC')}{gate_name}"

            # Update any key from ALL_QUBIT_KEYS that is present in gate_info
            # Additionally, if a gate is not controlled to begin with, we add the control indices
            for key in set(gate_info.keys()).intersection(ALL_QUBIT_KEYS):
                current_indices = gate_info.pop(key)

                gate_info[CONTROL_MAPPING[key]] = (
                    current_indices + num_controls if isinstance(current_indices, int)
                    else [index + num_controls for index in current_indices]
                )

            # If the gate is not controlled, we add the control indices
            control_indices = gate_info.get("control_indices", [])

            if isinstance(control_indices, int):
                control_indices = [control_indices]

            # Add control indices
            gate_info["control_indices"] = list(range(num_controls)) + \
                [idx for idx in control_indices if idx not in range(num_controls)]

            # Use the gate mapping to apply the corresponding gate with remaining kwargs
            # Add the control indices as the first indices given the number of control qubits
            getattr(controlled_circuit, gate_name)(**gate_info)

            # Re-insert gate name and definition into gate_info if needed elsewhere
            gate_info["gate"] = gate_name
            gate_info["definition"] = gate_definition

        if circuit.global_phase == 0:
            return controlled_circuit

        # Apply MCPhase on the control qubits to account for the global phase of the target gate
        # This essentially performs a relative phase correction to the control indices
        # If there is only one control qubit, we apply Phase gate, otherwise we apply MCPhase gate
        if num_controls == 1:
            controlled_circuit.Phase(circuit.global_phase, 0)
        else:
            controlled_circuit.MCPhase(
                circuit.global_phase,
                control_indices=list(range(num_controls))[:-1],
                target_indices=num_controls-1
            )

        return controlled_circuit

    def update(self) -> None:
        """ Update the circuit given the modified circuit log.

        Usage
        -----
        >>> circuit.update()
        """
        converted_circuit = self.convert(type(self))
        self.__dict__.update(converted_circuit.__dict__)

    @abstractmethod
    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:
        """ Convert the circuit to QASM.

        Parameters
        ----------
        `qasm_version` : int, optional
            The version of QASM to convert to. 2 for QASM 2.0 and 3 for QASM 3.0.

        Returns
        -------
        `qasm` : str
            The QASM representation of the circuit.

        Raises
        ------
        ValueError
            - QASM version must be either 2 or 3.

        Usage
        -----
        >>> circuit.to_qasm()
        """

    def to_quimb(
            self
        ) -> qtn.Circuit:
        """ Convert the circuit to Quimb Circuit. This method
        is used for performing tensor network simulations and
        optimizations using Quimb.

        Notes
        -----
        Quimb is a library for working with tensor networks in Python.
        It is built on top of NumPy and provides a high-level interface
        for working with tensor networks.

        The method will first transpile the circuit to U3 and CX gates
        before converting it to a Quimb Circuit.

        For more information, see the documentation:
        https://quimb.readthedocs.io/en/latest/

        Returns
        -------
        `quimb_circuit` : quimb.Circuit
            The Quimb representation of the circuit.

        Usage
        -----
        >>> quimb_circuit = circuit.to_quimb()
        """
        # Create a copy of the circuit as the `transpile` method is applied in-place
        circuit = self.copy()

        quimb_circuit = qtn.Circuit(N=self.num_qubits)

        circuit.transpile()

        for operation in circuit.circuit_log:
            if operation["gate"] == "U3":
                quimb_circuit.apply_gate(
                    gate_id="U3",
                    params=operation["angles"],
                    qubits=operation["target_indices"]
                )
            else:
                quimb_circuit.apply_gate(
                    gate_id=operation["CX"],
                    qubits=operation["target_indices"]
                )

        return quimb_circuit

    @staticmethod
    def from_cirq(
            cirq_circuit: cirq.Circuit,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Create a `qickit.Circuit` from a `cirq.Circuit`.

        Parameters
        ----------
        `cirq_circuit` : cirq.Circuit
            The Cirq quantum circuit to convert.
        `output_framework` : type[qickit.circuit.Circuit]
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circuit.Circuit`.

        Usage
        -----
        >>> circuit.from_cirq(cirq_circuit)
        """
        cirq_converter = FromCirq(output_framework=output_framework)
        circuit = cirq_converter.convert(cirq_circuit)
        return circuit

    @staticmethod
    def from_pennylane(
            pennylane_circuit: qml.QNode,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `qml.QNode`.

        Parameters
        ----------
        `pennylane_circuit` : qml.QNode
            The PennyLane quantum circuit to convert.
        `output_framework` : type[qickit.circuit.Circuit]
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circuit.Circuit`.

        Usage
        -----
        >>> circuit.from_pennylane(pennylane_circuit)
        """
        if not issubclass(output_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        # Define a circuit
        num_qubits = len(pennylane_circuit.device.wires)
        circuit = output_framework(num_qubits=num_qubits)

        # TODO: Implement the conversion from PennyLane to Qickit
        return circuit

    @staticmethod
    def from_qiskit(
            qiskit_circuit: qiskit.QuantumCircuit,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `qiskit.QuantumCircuit`.

        Parameters
        ----------
        `qiskit_circuit` : qiskit.QuantumCircuit
            The Qiskit quantum circuit to convert.
        `output_framework` : type[qickit.circuit.Circuit]
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circuit.Circuit`.

        Usage
        -----
        >>> converted_circuit = circuit.from_qiskit(qiskit_circuit)
        """
        qiskit_converter = FromQiskit(output_framework=output_framework)
        circuit = qiskit_converter.convert(qiskit_circuit)
        return circuit

    @staticmethod
    def from_tket(
            tket_circuit: pytket.Circuit,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `tket.Circuit`.

        Parameters
        ----------
        `tket_circuit` : tket.Circuit
            The TKET quantum circuit to convert.
        `output_framework` : type[qickit.circuit.Circuit]
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circuit.Circuit`.

        Usage
        -----
        >>> circuit.from_tket(tket_circuit)
        """
        tket_converter = FromTKET(output_framework=output_framework)
        circuit = tket_converter.convert(tket_circuit)
        return circuit

    @staticmethod
    def from_qasm(
            qasm: str,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a QASM string.

        Parameters
        ----------
        `qasm` : str
            The QASM string to convert.
        `output_framework` : type[qickit.circuit.Circuit]
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circuit.Circuit`.

        Usage
        -----
        >>> circuit.from_qasm(qasm)
        """
        if not issubclass(output_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        # Convert the QASM to qiskit circuit
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)

        # Convert the qiskit circuit to the output framework
        circuit = Circuit.from_qiskit(qiskit_circuit, output_framework)

        return circuit

    @staticmethod
    def from_quimb(
            quimb_circuit: qtn.Circuit,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Create a `qickit.circuit.Circuit` from a `qtn.Circuit`.

        Parameters
        ----------
        `quimb_circuit` : qtn.Circuit
            The Quimb quantum circuit to convert.
        `output_framework` : type[qickit.circuit.Circuit]
            The output framework to convert to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The converted circuit.

        Raises
        ------
        TypeError
            - The circuit framework must be a subclass of `qickit.circuit.Circuit`.

        Usage
        -----
        >>> circuit.from_quimb(quimb_circuit)
        """
        if not issubclass(output_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        num_qubits = 0
        circuit = output_framework(num_qubits=num_qubits)

        # TODO: Implement the conversion from Quimb to Qickit
        return circuit

    def copy(self) -> Circuit:
        """ Copy the circuit.

        Returns
        -------
        qickit.circuit.Circuit
            The copied circuit.

        Usage
        -----
        >>> copied_circuit = circuit.copy()
        """
        return copy.deepcopy(self)

    def reset(self) -> None:
        """ Reset the circuit to an empty circuit.

        Usage
        -----
        >>> circuit.reset()
        """
        self.__init__(num_qubits=self.num_qubits) # type: ignore

    @abstractmethod
    def draw(self):
        """ Draw the circuit.

        Usage
        -----
        >>> circuit.draw()
        """

    def plot_histogram(
            self,
            non_zeros_only: bool=False
        ) -> plt.Figure:
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
        plt.xlabel("State")
        plt.ylabel("Counts")
        plt.title("Histogram of the Circuit")
        plt.close()

        return figure

    def __eq__(
            self,
            other_circuit: object
        ) -> bool:
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
            - Circuits must be compared with other circuits.

        Usage
        -----
        >>> circuit1 == circuit2
        """
        if not isinstance(other_circuit, Circuit):
            raise TypeError("Circuits must be compared with other circuits.")
        return self.circuit_log == other_circuit.circuit_log

    def __len__(self) -> int:
        """ Get the number of the circuit operations.

        Returns
        -------
        int
            The number of the circuit operations.

        Usage
        -----
        >>> len(circuit)
        """
        return len(self.circuit_log)

    def __str__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        str
            The string representation of the circuit.

        Usage
        -----
        >>> str(circuit)
        """
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits})"

    def __repr__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        str
            The string representation of the circuit.

        Usage
        -----
        >>> repr(circuit)
        """
        repr_dict = []

        for operation in self.circuit_log:
            gate = {k:v for k,v in operation.items() if k != "definition"}
            repr_dict.append(gate)

        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, circuit_log={repr_dict})"

    def generate_calls(self) -> str:
        """ Generate the method calls by the circuit for
        reproducing the circuit. This method uses the circuit
        log to generate the method calls.

        Returns
        -------
        `calls` : str
            The method calls by the circuit.
        """
        calls = ""

        repr_dict = []

        for operation in self.circuit_log:
            gate = {k:v for k,v in operation.items() if k != "definition"}
            repr_dict.append(gate)

        for operation in repr_dict:
            gate = operation["gate"]
            args = ", ".join([f"{key}={value}" for key, value in operation.items() if key != "gate"])
            calls += f"circuit.{gate}({args})\n"

        return calls

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