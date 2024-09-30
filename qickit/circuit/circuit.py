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
import copy
import cmath
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from numpy.typing import NDArray
from types import NotImplementedType
from typing import Any, Literal, overload, SupportsFloat, SupportsIndex, Type, TYPE_CHECKING

import qiskit # type: ignore
import cirq # type: ignore
import pennylane as qml # type: ignore
import pytket
import pytket.circuit

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit.circuit_utils import (
    is_unitary_matrix, extract_rz, dec_uc_rotations, dec_ucg_help,
    simplify
)
from qickit.synthesis.unitarypreparation import (
    UnitaryPreparation, QiskitUnitaryTranspiler
)

EPSILON = 1e-10

""" Set the frozensets for the keys to be used:
- Decorator `Circuit.gatemethod()`
- Method `Circuit.vertical_reverse()`
- Method `Circuit.horizontal_reverse()`
- Method `Circuit.add()`
- Method `Circuit.change_mapping()`
"""
QUBIT_KEYS = frozenset(["qubit_index", "control_index", "target_index", "first_qubit_index",
                        "second_qubit_index", "first_target_index", "second_target_index"])
QUBIT_LIST_KEYS = frozenset(["qubit_indices", "control_indices", "target_indices"])
ANGLE_KEYS = frozenset(["angle", "angles"])
ALL_QUBIT_KEYS = QUBIT_KEYS.union(QUBIT_LIST_KEYS)


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
    `measured_qubits` : set[int]
        The set of measured qubits indices.
    `circuit_log` : list[dict]
        The log of the circuit operations.
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
        self.measured_qubits: set[int] = set()
        self.circuit_log: list[dict] = []
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
        match value:
            case range() | tuple() | Sequence():
                value = list(value)
            case np.ndarray():
                value = value.tolist()
            case SupportsIndex():
                value = int(value)
            case SupportsFloat():
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

        Raises
        ------
        TypeError
            - Qubit index must be an integer.
        ValueError
            - Qubit index out of range.
        """
        if name in ALL_QUBIT_KEYS:
            match value:
                case list():
                    if len(value) == 1:
                        value = value[0]

                        if not isinstance(value, int):
                            raise TypeError(f"Qubit index must be an integer. Unexpected type {type(value)} received.")

                        if value >= self.num_qubits or value < -self.num_qubits:
                            raise ValueError(f"Qubit index {value} out of range {self.num_qubits-1}.")

                        value = value if value >= 0 else self.num_qubits + value

                    else:
                        for i, index in enumerate(value):
                            if not isinstance(index, int):
                                raise TypeError(f"Qubit index must be an integer. Unexpected type {type(value)} received.")

                            if index >= self.num_qubits or index < -self.num_qubits:
                                raise ValueError(f"Qubit index {index} out of range {self.num_qubits-1}.")

                            value[i] = index if index >= 0 else self.num_qubits + index

                case int():
                    if value >= self.num_qubits or value < -self.num_qubits:
                        raise ValueError(f"Qubit index {value} out of range {self.num_qubits-1}.")

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
            match value:
                case list():
                    for angle in value:
                        if not isinstance(angle, (int, float)):
                            raise TypeError(f"Angle must be a number. Unexpected type {type(angle)} received.")
                        if np.isclose(angle, EPSILON) or np.isclose(angle % (2 * np.pi), EPSILON):
                            angle = 0
                    if all(angle == 0 for angle in value):
                        # Indicate no operation needed
                        return None
                case _:
                    if not isinstance(value, (int, float)):
                        raise TypeError(f"Angle must be a number. Unexpected type {type(value)} received.")
                    if np.isclose(value, EPSILON) or np.isclose(value % (2 * np.pi), EPSILON):
                        # Indicate no operation needed
                        return None

        return value

    def process_gate_params(
            self,
            gate: str,
            params: dict
        ) -> None:
        """ Process the gate parameters for the circuit.

        Parameters
        ----------
        `gate` : str
            The gate to apply to the circuit.
        `params` : dict
            The parameters of the gate.

        Usage
        -----
        >>> self.process_gate_params(gate="X", params={"qubit_indices": 0})
        """
        if not self.process_gate_params_flag:
            return

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
                return

            params[name] = value

        self.circuit_log.append({"gate": gate, **params})

    @abstractmethod
    def _single_qubit_gate(
            self,
            gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"],
            qubit_indices: int | Sequence[int],
            angle: float=0
        ) -> None:
        """ Apply a single qubit gate to the circuit.

        Parameters
        ----------
        `gate` : Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"]
            The gate to apply to the circuit.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.
        `angle` : float, optional, default=0
            The rotation angle in radians.

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
        >>> circuit._single_qubit_gate(gate="X", qubit_indices=0)
        >>> circuit._single_qubit_gate(gate="X", qubit_indices=[0, 1])
        >>> circuit._single_qubit_gate(gate="RX", qubit_indices=0, angle=np.pi/2)
        """

    @abstractmethod
    def _controlled_qubit_gate(
            self,
            gate: Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int],
            angle: float=0
        ) -> None:
        """ Apply a controlled gate to the circuit.

        Parameters
        ----------
        `gate` : Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"]
            The gate to apply to the circuit.
        `control_indices` : int | Collection[int]
            The index of the control qubit(s).
        `target_indices` : int | Collection[int]
            The index of the target qubit(s).
        `angle` : float, optional, default=0
            The rotation angle in radians.

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
        >>> circuit._non_parameterized_controlled_gate(gate="X", control_indices=0, target_indices=1)
        >>> circuit._non_parameterized_controlled_gate(gate="X", control_indices=[0, 1], target_indices=[2, 3])
        >>> circuit._parameterized_controlled_gate(gate="RX", angles=np.pi/2, control_indices=0, target_indices=1)
        >>> circuit._parameterized_controlled_gate(gate="RX", angles=np.pi/2, control_indices=[0, 1], target_indices=[2, 3])
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
        self.process_gate_params(gate=self.Identity.__name__, params=locals())
        self._single_qubit_gate(gate="I", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.X.__name__, params=locals())
        self._single_qubit_gate(gate="X", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.Y.__name__, params=locals())
        self._single_qubit_gate(gate="Y", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.Z.__name__, params=locals())
        self._single_qubit_gate(gate="Z", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.H.__name__, params=locals())
        self._single_qubit_gate(gate="H", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.S.__name__, params=locals())
        self._single_qubit_gate(gate="S", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.Sdg.__name__, params=locals())
        self._single_qubit_gate(gate="Sdg", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.T.__name__, params=locals())
        self._single_qubit_gate(gate="T", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.Tdg.__name__, params=locals())
        self._single_qubit_gate(gate="Tdg", qubit_indices=qubit_indices)

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
        self.process_gate_params(gate=self.RX.__name__, params=locals())
        self._single_qubit_gate(
            gate="RX",
            angle=angle,
            qubit_indices=qubit_indices
        )

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
        self.process_gate_params(gate=self.RY.__name__, params=locals())
        self._single_qubit_gate(
            gate="RY",
            angle=angle,
            qubit_indices=qubit_indices
        )

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
        self.process_gate_params(gate=self.RZ.__name__, params=locals())
        self._single_qubit_gate(
            gate="RZ",
            angle=angle,
            qubit_indices=qubit_indices
        )

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
        self.process_gate_params(gate=self.Phase.__name__, params=locals())
        self._single_qubit_gate(
            gate="Phase",
            angle=angle,
            qubit_indices=qubit_indices
        )

    @abstractmethod
    def U3(
            self,
            angles: Sequence[float],
            qubit_index: int
        ) -> None:
        """ Apply a U3 gate to the circuit.

        Parameters
        ----------
        `angles` : Sequence[float]
            The rotation angles in radians.
        `qubit_index` : int
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
        >>> circuit.U3(angles=[np.pi/2, np.pi/2, np.pi/2], qubit_index=0)
        """

    @abstractmethod
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
        self.process_gate_params(gate=self.CX.__name__, params=locals())
        self._controlled_qubit_gate(
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
        self.process_gate_params(gate=self.CY.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Y",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CZ.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Z",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CH.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="H",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CS.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="S",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CSdg.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Sdg",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CT.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="T",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CTdg.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Tdg",
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CRX.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="RX",
            angle=angle,
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CRY.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="RY",
            angle=angle,
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CRZ.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="RZ",
            angle=angle,
            control_indices=control_index,
            target_indices=target_index
        )

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
        self.process_gate_params(gate=self.CPhase.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Phase",
            angle=angle,
            control_indices=control_index,
            target_indices=target_index
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
        self.process_gate_params(gate=self.CU3.__name__, params=locals())
        self.MCU3(
            angles=angles,
            control_indices=control_index,
            target_indices=target_index
        )
        # Remove the last operation from the log to avoid duplication (This is to not add MCU3 to the log after CU3)
        if self.process_gate_params_flag:
            _ = self.circuit_log.pop()

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
        self.process_gate_params(gate=self.CSWAP.__name__, params=locals())
        self.MCSWAP(
            control_indices=control_index,
            first_target_index=first_target_index,
            second_target_index=second_target_index)
        # Remove the last operation from the log to avoid duplication (This is to not add MCSWAP to the log after CSWAP)
        if self.process_gate_params_flag:
            _ = self.circuit_log.pop()

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
        self.process_gate_params(gate=self.MCX.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="X",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCY.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Y",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCZ.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Z",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCH.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="H",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCS.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="S",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCSdg.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Sdg",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCT.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="T",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCTdg.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Tdg",
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCRX.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="RX",
            angle=angle,
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCRY.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="RY",
            angle=angle,
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCRZ.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="RZ",
            angle=angle,
            control_indices=control_indices,
            target_indices=target_indices
        )

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
        self.process_gate_params(gate=self.MCPhase.__name__, params=locals())
        self._controlled_qubit_gate(
            gate="Phase",
            angle=angle,
            control_indices=control_indices,
            target_indices=target_indices
        )

    @abstractmethod
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

    @abstractmethod
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

    def UCPauliRot(
        self,
        control_indices: int | Sequence[int],
        target_index: int,
        angles: Sequence[float],
        rot_axis: Literal["X", "Y", "Z"]
        ) -> None:
        """ Apply a uniformly controlled Pauli rotation to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.
        `angles` : Sequence[float]
            The rotation angles in radians.
        `rot_axis` : Literal["X", "Y", "Z"]
            The rotation axis.

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

        dec_uc_rotations(angles_params, 0, len(angles_params), False)

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
        control_indices: int | Sequence[int],
        target_index: int,
        angles: Sequence[float]
        ) -> None:
        """ Apply a uniformly controlled RX gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.
        `angles` : Sequence[float]
            The rotation angles in radians.

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
        >>> circuit.UCRX(control_indices=0, target_index=1, angles=[np.pi/2, np.pi/2])
        """
        self.UCPauliRot(control_indices, target_index, angles, "X")

    def UCRY(
        self,
        control_indices: int | Sequence[int],
        target_index: int,
        angles: Sequence[float]
        ) -> None:
        """ Apply a uniformly controlled RY gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.
        `angles` : Sequence[float]
            The rotation angles in radians.

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
        >>> circuit.UCRY(control_indices=0, target_index=1, angles=[np.pi/2, np.pi/2])
        """
        self.UCPauliRot(control_indices, target_index, angles, "Y")

    def UCRZ(
        self,
        control_indices: int | Sequence[int],
        target_index: int,
        angles: Sequence[float]
        ) -> None:
        """ Apply a uniformly controlled RZ gate to the circuit.

        Parameters
        ----------
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.
        `angles` : Sequence[float]
            The rotation angles in radians.

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
        >>> circuit.UCRZ(control_indices=0, target_index=1, angles=[np.pi/2, np.pi/2])
        """
        self.UCPauliRot(control_indices, target_index, angles, "Z")

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
            control_indices: int | Sequence[int],
            target_index: int,
            gates: list[NDArray[np.complex128]],
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
        `control_indices` : int | Sequence[int]
            The index of the control qubit(s).
        `target_index` : int
            The index of the target qubit.
        `gates` : list[NDArray[np.complex128]]
            The gates to apply to the circuit.
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
        (single_qubit_gates, diagonal) = dec_ucg_help(gates, len(control_indices) + 1)

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

    def unitary(
            self,
            unitary_matrix: NDArray[np.complex128],
            qubit_indices:  int | Sequence[int]
        ) -> None:
        """ Apply a unitary gate to the circuit. Uses Quantum Shannon Decomposition
        to decompose the unitary matrix into RY, RZ, and CX gates.

        Parameters
        ----------
        `unitary_matrix` : NDArray[np.complex128]
            The unitary matrix to apply to the circuit.
        `qubit_indices` : int | Sequence[int]
            The index of the qubit(s) to apply the gate to.

        Raises
        ------
        ValueError
            - The unitary matrix must have a size of 2^n x 2^n, where n is the number of qubits.
            - The unitary matrix must be unitary.
            - The number of qubits passed must be the same as the number of qubits needed to prepare the unitary.

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
        unitary_preparer = QiskitUnitaryTranspiler(output_framework=type(self))

        # Prepare the unitary matrix
        self = unitary_preparer.apply_unitary(
            circuit=self,
            unitary=unitary_matrix,
            qubit_indices=qubit_indices
        )

    def clbit_condition(
            self,
            clbit_index: int,
            clbit_value: int
        ) -> bool:
        """ Check if a classical bit meets a condition.

        Parameters
        ----------
        `clbit_index` : int
            The index of the classical bit to check.
        `clbit_value` : int
            The value to check the classical bit against.

        Returns
        -------
        `condition` : bool
            Whether the condition is met.

        Notes
        -----
        This method measures the classical bit at the specified index, and checks if it matches
        the specified value. This can be used to perform conditional operations in the circuit.

        Usage
        -----
        >>> if circuit.conditional(clbit_index=0, clbit_value=0):
        ...     circuit.X(qubit_indices=0)
        """
        # Measure the specified qubit
        self.measure(qubit_indices=clbit_index)

        for key, value in self.get_counts(num_shots=1).items():
            if value == 1:
                # Check if the value of the measurement matches the specified value
                condition = int(key[clbit_index]) == clbit_value
                break

        return condition

    def vertical_reverse(self) -> None:
        """ Perform a vertical reverse operation.

        Usage
        -----
        >>> circuit.vertical_reverse()
        """
        # Iterate over every operation, and change the index accordingly
        for operation in self.circuit_log:
            for key in set(operation.keys()).intersection(ALL_QUBIT_KEYS):
                match operation[key]:
                    case Sequence():
                        operation[key] = [(self.num_qubits - 1 - index) for index in operation[key]]
                    case _:
                        operation[key] = (self.num_qubits - 1 - operation[key])

        self.update()

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
        ValueError
            - The number of qubits must match the number of qubits in the `circuit`.

        Usage
        -----
        >>> circuit.add(circuit=circuit2, qubit_indices=0)
        >>> circuit.add(circuit=circuit2, qubit_indices=[0, 1])
        """
        if not isinstance(circuit, Circuit):
            raise TypeError("The circuit must be a Circuit object.")

        if isinstance(qubit_indices, Sequence):
            if len(qubit_indices) != circuit.num_qubits:
                raise ValueError("The number of qubits must match the number of qubits in the circuit.")

        # Create a copy of the as the `add` method is applied in-place
        circuit_log = copy.deepcopy(circuit.circuit_log)

        for operation in circuit_log:
            for key in set(operation.keys()).intersection(ALL_QUBIT_KEYS):
                match operation[key]:
                    case Sequence():
                        operation[key] = [qubit_indices[index] for index in operation[key]] # type: ignore
                    case _:
                        operation[key] = list(qubit_indices)[operation[key]] # type: ignore

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in circuit_log:
            # Extract gate name and remove it from gate_info for kwargs
            gate_name = gate_info.pop("gate", None)

            # Use the gate mapping to apply the corresponding gate with remaining kwargs
            getattr(self, gate_name)(**gate_info)

            # Re-insert gate name into gate_info if needed elsewhere
            gate_info["gate"] = gate_name

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

        Raises
        ------
        ValueError
            - If an index in `qubit_indices` has priorly been measured.

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
            if operation["gate"] == "measure":
                continue
            instructions.append(operation)

        return instructions

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

    @abstractmethod
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
            - Qubit indices must be a collection.
            - All qubit indices must be integers.
        ValueError
            - The number of qubits must match the number of qubits in the circuit.

        Usage
        -----
        >>> circuit.change_mapping(qubit_indices=[1, 0])
        """
        match qubit_indices:
            case Sequence():
                qubit_indices = list(qubit_indices)
            case np.ndarray():
                qubit_indices = qubit_indices.tolist()

        if not isinstance(qubit_indices, Sequence):
            raise TypeError("Qubit indices must be a collection.")

        if not all(isinstance(index, int) for index in qubit_indices):
            raise TypeError("All qubit indices must be integers.")

        if self.num_qubits != len(qubit_indices):
            raise ValueError("The number of qubits must match the number of qubits in the circuit.")

        # Update the qubit indices
        for operation in self.circuit_log:
            for key in set(operation.keys()).intersection(ALL_QUBIT_KEYS):
                match operation[key]:
                    case list():
                        operation[key] = [qubit_indices[index] for index in operation[key]]
                    case _:
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

        # Set the circuit log of the new circuit to the current circuit log
        # We do this so that we can disable the processing of gate parameters
        # and directly apply the gates in the new framework
        converted_circuit.circuit_log = self.circuit_log
        converted_circuit.process_gate_params_flag = False

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in self.circuit_log:
            # Extract gate name and remove it from gate_info for kwargs
            gate_name = gate_info.pop("gate", None)

            # Use the gate mapping to apply the corresponding gate with remaining kwargs
            getattr(converted_circuit, gate_name)(**gate_info)

            # Re-insert gate name into gate_info if needed elsewhere
            gate_info["gate"] = gate_name

        # Re-enable the processing of gate parameters
        converted_circuit.process_gate_params_flag = True

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

        # Define a controlled circuit
        controlled_circuit = type(circuit)(num_qubits=circuit.num_qubits + num_controls)

        # Iterate over the gate log and apply corresponding gates in the new framework
        for gate_info in circuit.circuit_log:
            # Extract gate name and remove it from gate_info for kwargs
            gate_name = gate_info.pop("gate", None)

            # Change the gate name from single qubit and controlled to multi-controlled
            match gate_name[0]:
                case "C":
                    gate_name = f"M{gate_name}"
                case "M":
                    pass
                case _:
                    gate_name = f"MC{gate_name}"

            if not any(key in gate_info for key in ["control_index", "control_indices"]):
                # For single qubit gates
                if "qubit_indices" in gate_info:
                    qubit_indices = gate_info.pop("qubit_indices", None)
                    if isinstance(qubit_indices, int):
                        qubit_indices = [qubit_indices]
                    gate_info["target_indices"] = [qubit_index + num_controls for qubit_index in qubit_indices]
                    gate_info["control_indices"] = []

                # For U3 gate
                elif "qubit_index" in gate_info:
                    gate_info["target_indices"] = gate_info.pop("qubit_index", None) + num_controls
                    gate_info["control_indices"] = []

                # For SWAP gate
                elif "first_qubit_index" in gate_info:
                    gate_info["first_target_index"] = gate_info.pop("first_qubit_index", None) + num_controls
                    gate_info["second_target_index"] = gate_info.pop("second_qubit_index", None) + num_controls
                    gate_info["control_indices"] = []

            else:
                # For controlled gates
                if "target_indices" in gate_info:
                    target_indices = gate_info.pop("target_indices", None)
                    control_indices = gate_info.pop("control_indices", None)
                    if isinstance(target_indices, int):
                        target_indices = [target_indices]
                    gate_info["target_indices"] = [target_index + num_controls for target_index in target_indices]
                    gate_info["control_indices"] = [control_index + num_controls for control_index in control_indices]

                # For single-controlled gates
                elif "target_index" in gate_info:
                    gate_info["target_indices"] = gate_info.pop("target_index", None) + num_controls
                    gate_info["control_indices"] = [gate_info.pop("control_index", None) + num_controls]

                # For CSWAP and MCSWAP gates
                elif "first_target_index" in gate_info:
                    gate_info["first_target_index"] = gate_info.pop("first_target_index", None) + num_controls
                    gate_info["second_target_index"] = gate_info.pop("second_target_index", None) + num_controls
                    if "control_indices" in gate_info:
                        control_indices = gate_info.pop("control_indices", None)
                        gate_info["control_indices"] = [control_index + num_controls for control_index in control_indices]
                    else:
                        gate_info["control_indices"] = [gate_info.pop("control_index", None) + num_controls]

            gate_info["control_indices"] = list(range(num_controls)) + gate_info.pop("control_indices", None)

            # Use the gate mapping to apply the corresponding gate with remaining kwargs
            # Add the control indices as the first indices given the number of control qubits
            getattr(controlled_circuit, gate_name)(**gate_info)

            # Re-insert gate name into gate_info if needed elsewhere
            gate_info["gate"] = gate_name

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
        if not issubclass(output_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        # Define a circuit
        num_qubits = len(cirq_circuit.all_qubits())
        circuit = output_framework(num_qubits=num_qubits)

        # Define the list of all circuit operations
        ops = list(cirq_circuit.all_operations())

        # Iterate over the operations in the Cirq circuit
        for operation in ops:
            gate = operation.gate
            gate_type = type(gate).__name__

            qubits = operation.qubits
            if gate_type != "GlobalPhaseGate":
                qubit_indices = [qubit.x for qubit in qubits] if len(qubits) > 1 else qubits[0].x # type: ignore

            # Extract the parameters of the gate
            parameters = gate._json_dict_() # type: ignore

            # TODO: Add U3, CU3, and MCU3 support (Note: Cirq doesn't have built-in U3 gate)
            if gate_type == "IdentityGate":
                circuit.Identity(qubit_indices)

            elif gate_type == "_PauliX":
                circuit.X(qubit_indices)

            elif gate_type == "_PauliY":
                circuit.Y(qubit_indices)

            elif gate_type == "_PauliZ":
                circuit.Z(qubit_indices)

            elif gate_type == "HPowGate":
                circuit.H(qubit_indices)

            elif gate_type == "ZPowGate":
                if parameters["exponent"] == 0.5:
                    circuit.S(qubit_indices)
                elif parameters["exponent"] == 0.25:
                    circuit.T(qubit_indices)
                elif parameters["exponent"] == -0.5:
                    circuit.Sdg(qubit_indices)
                elif parameters["exponent"] == -0.25:
                    circuit.Tdg(qubit_indices)
                else:
                    circuit.Phase(parameters["exponent"], qubit_indices)

                    if parameters["global_shift"] != 0:
                        global_phase_angle = abs(
                            np.log(parameters["global_shift"])
                        )
                        circuit.GlobalPhase(global_phase_angle)

            elif gate_type == "S":
                circuit.S(qubit_indices)

            elif gate_type == "T":
                circuit.T(qubit_indices)

            elif gate_type == "Rx":
                if isinstance(qubit_indices, list):
                    for qubit_index in qubit_indices:
                        circuit.RX(parameters["rads"], qubit_index)
                else:
                    circuit.RX(parameters["rads"], qubit_indices)

            elif gate_type == "Ry":
                if isinstance(qubit_indices, list):
                    for qubit_index in qubit_indices:
                        circuit.RY(parameters["rads"], qubit_index)
                else:
                    circuit.RY(parameters["rads"], qubit_indices)

            elif gate_type == "Rz":
                if isinstance(qubit_indices, list):
                    for qubit_index in qubit_indices:
                        circuit.RZ(parameters["rads"], qubit_index)
                else:
                    circuit.RZ(parameters["rads"], qubit_indices)

            elif gate_type == "SwapPowGate":
                circuit.SWAP(qubit_indices[0], qubit_indices[1])

            elif gate_type == "GlobalPhaseGate":
                global_phase_angle = abs(
                    np.log(parameters["coefficient"])
                )
                circuit.GlobalPhase(global_phase_angle)

            elif gate_type == "MeasurementGate":
                circuit.measure(qubit_indices)

            elif gate_type == "ControlledGate":
                if parameters["sub_gate"] == cirq.X:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCX(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CX(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.Y:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCY(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CY(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.Z:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCZ(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CZ(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.H:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCH(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CH(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.S:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCS(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CS(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.S**-1:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCSdg(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CSdg(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.T:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCT(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CT(qubit_indices[0], qubit_indices[1])

                elif parameters["sub_gate"] == cirq.T**-1:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCTdg(
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CTdg(qubit_indices[0], qubit_indices[1])

                elif isinstance(parameters["sub_gate"], cirq.Rx):
                    angle = parameters["sub_gate"]._json_dict_()["rads"]
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCRX(
                            angle,
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CRX(angle, qubit_indices[0], qubit_indices[1])

                elif isinstance(parameters["sub_gate"], cirq.Ry):
                    angle = parameters["sub_gate"]._json_dict_()["rads"]
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCRY(
                            angle,
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CRY(angle, qubit_indices[0], qubit_indices[1])

                elif isinstance(parameters["sub_gate"], cirq.Rz):
                    angle = parameters["sub_gate"]._json_dict_()["rads"]
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCRZ(
                            angle,
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CRZ(angle, qubit_indices[0], qubit_indices[1])

                elif isinstance(parameters["sub_gate"], cirq.ZPowGate):
                    angle = parameters["sub_gate"]._json_dict_()["exponent"]
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCPhase(
                            angle,
                            control_indices=qubit_indices[:-1],
                            target_indices=qubit_indices[-1]
                        )
                    else:
                        circuit.CPhase(angle, qubit_indices[0], qubit_indices[1])

                    if parameters["sub_gate"]._json_dict_()["global_shift"] != 0:
                        global_phase_angle = abs(
                            np.log(parameters["sub_gate"]._json_dict_()["global_shift"])
                        )
                        circuit.GlobalPhase(global_phase_angle)

                elif parameters["sub_gate"] == cirq.SWAP:
                    if len(parameters["control_qid_shape"]) > 1:
                        circuit.MCSWAP(
                            control_indices=qubit_indices[:-2],
                            first_target_index=qubit_indices[-2],
                            second_target_index=qubit_indices[-1]
                        )
                    else:
                        circuit.CSWAP(qubit_indices[0], qubit_indices[1], qubit_indices[2])

            else:
                raise ValueError(f"Gate not supported.\n{operation} ")

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
        >>> circuit.from_qiskit(qiskit_circuit)
        """
        if not issubclass(output_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        def match_pattern(
                string: str,
                gate_name: str
            ) -> bool:
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
            prefix = (f"c{gate_name}", f"cc{gate_name}", f"mc{gate_name}")
            if string in prefix:
                return True
            return False

        # Define a circuit
        num_qubits = qiskit_circuit.num_qubits
        circuit = output_framework(num_qubits=num_qubits)

        # Define the gates to skip
        skip_gates = ["barrier", "global_phase"]

        # Iterate over the operations in the Qiskit circuit
        for gate in qiskit_circuit.data:
            gate_type = gate.operation.name

            if gate_type not in skip_gates:
                qubit_indices = [int(qubit._index) for qubit in gate.qubits] if len(gate.qubits) > 1 else [int(gate.qubits[0]._index)]
            elif gate_type == "global_phase":
                pass
            else:
                continue

            if gate_type == "id":
                circuit.Identity(qubit_indices)

            elif gate_type == "x":
                circuit.X(qubit_indices)

            elif gate_type == "y":
                circuit.Y(qubit_indices)

            elif gate_type == "z":
                circuit.Z(qubit_indices)

            elif gate_type == "h":
                circuit.H(qubit_indices)

            elif gate_type == "s":
                circuit.S(qubit_indices)

            elif gate_type == "sdg":
                circuit.Sdg(qubit_indices)

            elif gate_type == "t":
                circuit.T(qubit_indices)

            elif gate_type == "tdg":
                circuit.Tdg(qubit_indices)

            elif gate_type == "rx":
                circuit.RX(gate.operation.params[0], qubit_indices)

            elif gate_type == "ry":
                circuit.RY(gate.operation.params[0], qubit_indices)

            elif gate_type == "rz":
                circuit.RZ(gate.operation.params[0], qubit_indices)

            elif gate_type == "p":
                circuit.Phase(gate.operation.params[0], qubit_indices)

            elif gate_type in ["u", "u3"]:
                circuit.U3(gate.operation.params, qubit_indices[0])

            elif gate_type == "swap":
                circuit.SWAP(qubit_indices[0], qubit_indices[1])

            elif gate_type == "cx":
                circuit.CX(qubit_indices[0], qubit_indices[1])

            elif gate_type == "cy":
                circuit.CY(qubit_indices[0], qubit_indices[1])

            elif gate_type == "cz":
                circuit.CZ(qubit_indices[0], qubit_indices[1])

            elif gate_type == "ch":
                circuit.CH(qubit_indices[0], qubit_indices[1])

            elif gate_type == "cs":
                circuit.CS(qubit_indices[0], qubit_indices[1])

            elif gate_type == "csdg":
                circuit.CSdg(qubit_indices[0], qubit_indices[1])

            elif gate_type == "ct":
                circuit.CT(qubit_indices[0], qubit_indices[1])

            elif gate_type == "ctdg":
                circuit.CTdg(qubit_indices[0], qubit_indices[1])

            elif gate_type == "crx":
                circuit.CRX(gate.operation.params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == "cry":
                circuit.CRY(gate.operation.params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == "crz":
                circuit.CRZ(gate.operation.params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == "cp":
                circuit.CPhase(gate.operation.params[0], qubit_indices[0], qubit_indices[1])

            elif gate_type == "cu3":
                circuit.CU3(gate.operation.params, qubit_indices[0], qubit_indices[1])

            elif gate_type == "cswap":
                circuit.CSWAP(qubit_indices[0], qubit_indices[1], qubit_indices[2])

            elif gate_type == "measure":
                circuit.measure(qubit_indices)

            elif gate_type == "global_phase":
                circuit.GlobalPhase(gate.operation.params[0])

            elif match_pattern(gate_type, "x"):
                circuit.MCX(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "y"):
                circuit.MCY(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "z"):
                circuit.MCZ(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "h"):
                circuit.MCH(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "s"):
                circuit.MCS(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "sdg"):
                circuit.MCSdg(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "t"):
                circuit.MCT(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "tdg"):
                circuit.MCTdg(qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "rx"):
                circuit.MCRX(gate.operation.params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "ry"):
                circuit.MCRY(gate.operation.params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "rz"):
                circuit.MCRZ(gate.operation.params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "phase"):
                circuit.MCPhase(gate.operation.params[0], qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "u3"):
                circuit.MCU3(gate.operation.params, qubit_indices[:-1], qubit_indices[-1])

            elif match_pattern(gate_type, "swap"):
                circuit.MCSWAP(qubit_indices[:-2], qubit_indices[-2], qubit_indices[-1])

            else:
                raise ValueError(f"Gate not supported.\n{gate_type} ")

        # Apply the global phase of the `qiskit_circuit`
        circuit.GlobalPhase(qiskit_circuit.global_phase)

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
        if not issubclass(output_framework, Circuit):
            raise TypeError("The circuit framework must be a subclass of `qickit.circuit.Circuit`.")

        # Define a circuit
        num_qubits = tket_circuit.n_qubits
        circuit = output_framework(num_qubits=num_qubits)

        # Iterate over the operations in the Qiskit circuit
        for gate in tket_circuit:
            gate_type = str(gate.op.type)
            qubit_indices = [int(qubit.index[0]) for qubit in gate.qubits] if len(gate.qubits) > 1 \
                                                                           else [gate.qubits[0].index[0]]

            if gate_type == "OpType.noop":
                circuit.Identity(qubit_indices)

            elif gate_type == "OpType.X":
                circuit.X(qubit_indices)

            elif gate_type == "OpType.Y":
                circuit.Y(qubit_indices)

            elif gate_type == "OpType.Z":
                circuit.Z(qubit_indices)

            elif gate_type == "OpType.H":
                circuit.H(qubit_indices)

            elif gate_type == "OpType.S":
                circuit.S(qubit_indices)

            elif gate_type == "OpType.Sdg":
                circuit.Sdg(qubit_indices)

            elif gate_type == "OpType.T":
                circuit.T(qubit_indices)

            elif gate_type == "OpType.Tdg":
                circuit.Tdg(qubit_indices)

            elif gate_type == "OpType.Rx":
                circuit.RX(float(gate.op.params[0]), qubit_indices[0])

            elif gate_type == "OpType.Ry":
                circuit.RY(float(gate.op.params[0]), qubit_indices[0])

            elif gate_type == "OpType.Rz":
                circuit.RZ(float(gate.op.params[0]), qubit_indices[0])

            elif gate_type == "OpType.U1":
                circuit.Phase(float(gate.op.params[0]), qubit_indices[0])

            elif gate_type == "OpType.U3":
                circuit.U3([float(param) for param in gate.op.params], qubit_indices[0])

            elif gate_type == "OpType.SWAP":
                circuit.SWAP(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CX":
                circuit.CX(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CY":
                circuit.CY(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CZ":
                circuit.CZ(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CH":
                circuit.CH(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CS":
                circuit.CS(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CSdg":
                circuit.CSdg(qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CRx":
                circuit.CRX(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CRy":
                circuit.CRY(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CRz":
                circuit.CRZ(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CU1":
                circuit.CPhase(float(gate.op.params[0]), qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CU3":
                circuit.CU3([float(param) for param in gate.op.params], qubit_indices[0], qubit_indices[1])

            elif gate_type == "OpType.CSWAP":
                circuit.CSWAP(qubit_indices[0], qubit_indices[1], qubit_indices[2])

            elif gate_type == "OpType.CnX":
                circuit.MCX(qubit_indices[:-1], qubit_indices[-1])

            elif gate_type == "OpType.CnY":
                circuit.MCY(qubit_indices[:-1], qubit_indices[-1])

            elif gate_type == "OpType.CnZ":
                circuit.MCZ(qubit_indices[:-1], qubit_indices[-1])

            elif gate_type == "OpType.Measure":
                circuit.measure(qubit_indices)

            elif isinstance(gate.op, pytket.circuit.QControlBox):
                qcontrolbox = gate.op

                if "X" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCX(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CX(qubit_indices[0], qubit_indices[1])

                elif "Y" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCY(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CY(qubit_indices[0], qubit_indices[1])

                elif "Z" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCZ(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CZ(qubit_indices[0], qubit_indices[1])

                elif "H" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCH(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CH(qubit_indices[0], qubit_indices[1])

                elif "SWAP" in str(qcontrolbox.get_op()):
                    circuit.MCSWAP(qubit_indices[:-2], qubit_indices[-2], qubit_indices[-1])

                elif "Sdg" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCSdg(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CSdg(qubit_indices[0], qubit_indices[1])

                elif "S" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCS(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CS(qubit_indices[0], qubit_indices[1])

                elif "Tdg" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCTdg(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CTdg(qubit_indices[0], qubit_indices[1])

                elif "T" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCT(qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CT(qubit_indices[0], qubit_indices[1])

                elif "Rx" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCRX(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CRX(float(gate.op.get_op().params[0]), qubit_indices[0], qubit_indices[1])

                elif "Ry" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCRY(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CRY(float(gate.op.get_op().params[0]), qubit_indices[0], qubit_indices[1])

                elif "Rz" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCRZ(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CRZ(float(gate.op.get_op().params[0]), qubit_indices[0], qubit_indices[1])

                elif "U1" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCPhase(float(gate.op.get_op().params[0]), qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CPhase(float(gate.op.get_op().params[0]), qubit_indices[0], qubit_indices[1])

                elif "U3" in str(qcontrolbox.get_op()):
                    if len(qubit_indices) > 2:
                        circuit.MCU3([float(param) for param in gate.op.get_op().params], qubit_indices[:-1], qubit_indices[-1])
                    else:
                        circuit.CU3([float(param) for param in gate.op.get_op().params], qubit_indices[0], qubit_indices[1])

            else:
                raise ValueError(f"Gate not supported.\n{gate_type} ")

        # Apply the global phase of the `tket_circuit`
        circuit.GlobalPhase(float(tket_circuit.phase))

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

        # Define a circuit
        num_qubits = 0
        circuit = output_framework(num_qubits=num_qubits)

        # TODO: Implement the conversion from QASM to Qickit
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
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, circuit_log={self.circuit_log})"

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