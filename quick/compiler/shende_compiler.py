# Copyright 2023-2025 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/quick/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Compiler class for compiling primitives into circuits.
"""

from __future__ import annotations

__all__ = ["ShendeCompiler"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

from quick.circuit import Circuit
from quick.primitives import Statevector, Operator
from quick.synthesis.statepreparation import StatePreparation, Isometry
from quick.synthesis.unitarypreparation import UnitaryPreparation, ShannonDecomposition

""" Type aliases for the primitives to be compiled:
- `PRIMITIVE` is a single primitive object, which can be a `Statevector`, `Operator`,or a `numpy.ndarray`.
- `PRIMITIVES` is a list of tuples containing the primitive object and the qubits they need to be applied to.
"""
PRIMITIVE: TypeAlias = Statevector | Operator | NDArray[np.complex128]
PRIMITIVES: TypeAlias = list[tuple[PRIMITIVE, Sequence[int]]]


class ShendeCompiler:
    """ `quick.compiler.ShendeCompiler` provides compilation of states and unitaries
    into circuits using the Shende et al. method. The `compile` method is the main
    interface for the compiler, which takes in a primitives object and returns a
    circuit object.

    Notes
    -----
    To create a custom compiler, subclass `quick.compiler.ShendeCompiler` and overwrite the
    `state_preparation`, `unitary_preparation`, and `compile` methods. The default compiler
    uses Shende et al for compilation.

    - Publication:
    https://arxiv.org/abs/quant-ph/0406176
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    Parameters
    ----------
    `circuit_framework` : type[quick.circuit.Circuit]
        The circuit framework for the compiler.
    `state_prep` : type[quick.synthesis.statepreparation.StatePreparation], optional, default=Shende
        The state preparation schema for the compiler. Use `Shende` for the default schema.
    `unitary_prep` : type[quick.synthesis.unitarypreparation.UnitaryPreparation], optional, default=ShannonDecomposition
        The unitary preparation schema for the compiler. Use `ShannonDecomposition` for the default schema.

    Attributes
    ----------
    `circuit_framework` : type[quick.circuit.Circuit]
        The circuit framework for the compiler.
    `state_prep` : quick.synthesis.statepreparation.StatePreparation
        The state preparation schema for the compiler.
    `unitary_prep` : quick.synthesis.unitarypreparation.UnitaryPreparation
        The unitary preparation schema for the compiler.

    Raises
    ------
    TypeError
        - If the circuit framework is invalid.
        - If the state preparation schema is invalid.
        - If the unitary preparation schema is invalid.
        - If the optimizer is invalid.

    Usage
    -----
    >>> compiler = Compiler(circuit_framework, state_prep, unitary_prep, mlir)
    >>> circuit = compiler.compile(primitives)
    """
    def __init__(
            self,
            circuit_framework: type[Circuit],
            state_prep: type[StatePreparation] = Isometry,
            unitary_prep: type[UnitaryPreparation] = ShannonDecomposition,
        ) -> None:
        """ Initialize a `quick.compiler.Compiler` object.
        """
        if not issubclass(circuit_framework, Circuit):
            raise TypeError("Invalid circuit framework.")
        if not issubclass(state_prep, StatePreparation):
            raise TypeError("Invalid state preparation schema.")
        if not issubclass(unitary_prep, UnitaryPreparation):
            raise TypeError("Invalid unitary preparation schema.")

        self.circuit_framework = circuit_framework
        self.state_prep = state_prep(circuit_framework)
        self.unitary_prep = unitary_prep(circuit_framework)

    def state_preparation(
            self,
            state: NDArray[np.complex128] | Statevector,
        ) -> Circuit:
        """ Prepare a quantum state.

        Parameters
        ----------
        `state` : NDArray[np.complex128] | quick.primitives.Statevector
            The quantum state to be prepared.

        Returns
        -------
        `quick.circuit.Circuit`
            The circuit object for the state preparation.

        Usage
        -----
        >>> circuit = compiler.state_preparation(state)
        """
        return self.state_prep.prepare_state(state)

    def unitary_preparation(
            self,
            unitary: NDArray[np.complex128] | Operator
        ) -> Circuit:
        """ Prepare a quantum unitary.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128] | quick.primitives.Operator
            The quantum unitary to be prepared.

        Returns
        -------
        `quick.circuit.Circuit`
            The circuit object for the unitary preparation.

        Usage
        -----
        >>> circuit = compiler.unitary_preparation(unitary)
        """
        return self.unitary_prep.prepare_unitary(unitary)

    @staticmethod
    def _check_primitive(primitive: PRIMITIVE) -> None:
        """ Check if the primitive object is valid.

        Parameters
        ----------
        `primitive` : PRIMITIVE
            The primitive object to be checked.

        Raises
        ------
        ValueError
            - If the primitive object is invalid.
        """
        if not isinstance(primitive, (Statevector, Operator, np.ndarray)):
            raise TypeError("Invalid primitive object.")

        if isinstance(primitive, np.ndarray):
            if len(primitive.ravel()) < 2:
                raise ValueError("Invalid primitive object.")
            elif primitive.ndim not in [1, 2]:
                raise ValueError("Invalid primitive object.")
            elif primitive.ndim == 2:
                if not primitive.shape[0] == primitive.shape[1]:
                    raise ValueError("Invalid primitives object.")

    @staticmethod
    def _check_primitive_qubits(
            primitive: PRIMITIVE,
            qubit_indices: Sequence[int]
        ) -> None:
        """ Check if the primitives object is valid. The primitives object should be a list of

        Parameters
        ----------
        `primitives` : PRIMITIVE
            The primitives object to be checked.
        `qubit_indices` : Sequence[int]
            The list of qubit indices to apply the primitive to.

        Raises
        ------
        ValueError
            - The number of qubits should be the same as the number of qubits required for
            preparing the primitive object.
        """
        if isinstance(primitive, np.ndarray):
            if not 2**len(qubit_indices) == primitive.shape[0]:
                raise ValueError("The number of qubits should be the same as the number of qubits required.")
        elif not len(qubit_indices) == primitive.num_qubits:
            raise ValueError("The number of qubits should be the same as the number of qubits required.")

    @staticmethod
    def _check_primitives(primitives: PRIMITIVES) -> None:
        """ Check if the primitives object is valid. The primitives object should be a list of
        tuples containing the primitive object and the qubits they need to be applied to.
        Furthermore, the list of qubits should be the same length as the number of qubits
        required for preparing the primitive object.

        Parameters
        ----------
        `primitives` : PRIMITIVES
            The primitives object to be checked. A list of tuples containing the primitives
            object and the qubits they need to be applied to.

        Raises
        ------
        ValueError
            - If the primitives object is invalid.
            - The number of qubits should be the same as the number of qubits required for
            preparing the primitive object.
        """
        for primitive, qubit_indices in primitives:
            ShendeCompiler._check_primitive(primitive)
            ShendeCompiler._check_primitive_qubits(primitive, qubit_indices)

    def _compile_primitive(
            self,
            primitive: PRIMITIVE,
        ) -> Circuit:
        """ Compile a single primitive object into a circuit object.

        Parameters
        ----------
        `primitive` : PRIMITIVE
            The primitive object to be compiled.

        Returns
        -------
        `quick.circuit.Circuit`
            The compiled circuit object.

        Raises
        ------
        ValueError
            - If the primitive object is invalid.
        """
        self._check_primitive(primitive)

        if isinstance(primitive, Statevector):
            return self.state_preparation(primitive)
        elif isinstance(primitive, Operator):
            return self.unitary_preparation(primitive)
        elif isinstance(primitive, np.ndarray):
            if primitive.ndim == 1:
                return self.state_preparation(Statevector(primitive))
            else:
                return self.unitary_preparation(Operator(primitive))

    def compile(
            self,
            primitives: PRIMITIVE | PRIMITIVES
        ) -> Circuit:
        """ Compile the primitives object into a circuit object.

        Parameters
        ----------
        `primitives` : PRIMITIVE | PRIMITIVES
            The primitives object to be compiled. Alternatively, a list of tuples
            containing the primitives object and the qubits they need to be applied
            to can be provided to compile multiple primitives into one circuit in
            the order which they have been defined.

        Returns
        -------
        `quick.circuit.Circuit`
            The compiled circuit object.

        Raises
        ------
        ValueError
            - If the primitives object is invalid.
            - The number of qubits should be the same as the number of qubits required for
            preparing the primitive object.

        Usage
        -----
        >>> primitive = Operator(unitary_matrix)
        >>> circuit = compiler.compile(primitive)
        >>> primitives = [(Operator(unitary_matrix), [0, 1]),
        ...               (Statevector(statevector), [2, 3])]
        >>> circuit = compiler.compile(primitives)
        """
        if isinstance(primitives, list):
            self._check_primitives(primitives)
        else:
            return self._compile_primitive(primitives)

        max_qubits = 0

        for _, qubits in primitives:
            if max(qubits) > max_qubits:
                max_qubits = max(qubits)

        circuit = self.circuit_framework(max_qubits + 1)

        for primitive, qubits in primitives:
            compiled_circuit = self._compile_primitive(primitive)
            circuit.add(compiled_circuit, qubits)

        return circuit