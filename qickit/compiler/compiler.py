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

__all__ = ["Compiler"]

import numpy as np
from numpy.typing import NDArray
from typing import Type, TypeAlias

from qickit.circuit import Circuit
from qickit.primitives import Bra, Ket, Operator
from qickit.synthesis.statepreparation import StatePreparation, Shende
from qickit.synthesis.unitarypreparation import UnitaryPreparation, ShannonDecomposition

# Define the type alias for the primitives object
PRIMITIVE: TypeAlias = Bra | Ket | Operator | NDArray[np.complex128]


class Compiler:
    """ `qickit.compiler.Compiler` is the base class for creating quantum compilation passes
    from primitives to circuits. The `compile` method is the main interface for the compiler,
    which takes in a primitives object and returns a circuit object. The default compiler uses
    Shende et al for compilation.
    ref: https://arxiv.org/abs/quant-ph/0406176

    Notes
    -----
    To create a custom compiler, subclass `qickit.compiler.Compiler` and overwrite the
    `state_preparation`, `unitary_preparation`, and `compile` methods.

    Parameters
    ----------
    `circuit_framework` : qickit.circuit.Circuit
        The circuit framework for the compiler.
    `state_prep` : type[qickit.synthesis.statepreparation.StatePreparation], optional, default=Shende
        The state preparation schema for the compiler. Use `Shende` for the default schema.
    `unitary_prep` : type[qickit.synthesis.unitarypreparation.UnitaryPreparation], optional, default=ShannonDecomposition
        The unitary preparation schema for the compiler. Use `ShannonDecomposition` for the default schema.

    Attributes
    ----------
    `circuit_framework` : qickit.circuit.Circuit
        The circuit framework for the compiler.
    `state_prep` : qickit.synthesis.statepreparation.StatePreparation
        The state preparation schema for the compiler.
    `unitary_prep` : qickit.synthesis.unitarypreparation.UnitaryPreparation
        The unitary preparation schema for the compiler.

    Raises
    ------
    ValueError
        If the circuit framework is invalid.
        If the state preparation schema is invalid.
        If the unitary preparation schema is invalid.

    Usage
    -----
    >>> compiler = Compiler(circuit_framework, state_prep, unitary_prep, mlir)
    >>> circuit = compiler.compile(primitives)
    """
    def __init__(
            self,
            circuit_framework: Type[Circuit],
            state_prep: Type[StatePreparation]=Shende,
            unitary_prep: Type[UnitaryPreparation]=ShannonDecomposition,
            mlir: bool=True
        ) -> None:
        """ Initialize a `qickit.compiler.Compiler` object.
        """
        if not issubclass(circuit_framework, Circuit):
            raise ValueError("Invalid circuit framework.")
        if not issubclass(state_prep, StatePreparation):
            raise ValueError("Invalid state preparation schema.")
        if not issubclass(unitary_prep, UnitaryPreparation):
            raise ValueError("Invalid unitary preparation schema.")

        self.circuit_framework = circuit_framework
        self.state_prep = state_prep(circuit_framework)
        self.unitary_prep = unitary_prep(circuit_framework)

    def state_preparation(
            self,
            state: NDArray[np.complex128] | Bra | Ket,
        ) -> Circuit:
        """ Prepare a quantum state.

        Parameters
        ----------
        `state` : NDArray[np.complex128] | qickit.primitives.Bra | qickit.primitives.Ket
            The quantum state to be prepared.

        Returns
        -------
        `qickit.circuit.Circuit`
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
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The quantum unitary to be prepared.

        Returns
        -------
        `qickit.circuit.Circuit`
            The circuit object for the unitary preparation.

        Usage
        -----
        >>> circuit = compiler.unitary_preparation(unitary)
        """
        return self.unitary_prep.prepare_unitary(unitary)

    def compile(
        self,
        primitives: PRIMITIVE
        ) -> Circuit:
        """ Compile the primitives object into a circuit object.

        Parameters
        ----------
        `primitives` : PRIMITIVE
            The primitives object to be compiled.

        Returns
        -------
        `qickit.circuit.Circuit`
            The compiled circuit object.

        Raises
        ------
        ValueError
            If the primitives object is invalid.

        Usage
        -----
        >>> circuit = compiler.compile(primitives)
        """
        match primitives:
            case Bra():
                return self.state_preparation(primitives)
            case Ket():
                return self.state_preparation(primitives)
            case Operator():
                return self.unitary_preparation(primitives)
            case np.ndarray():
                if primitives.ndim == 1:
                    return self.state_preparation(Ket(primitives))
                elif primitives.ndim == 2:
                    if primitives.shape[0] == primitives.shape[1]:
                        return self.unitary_preparation(Operator(primitives))
                    elif primitives.shape[1] == 1:
                        return self.state_preparation(Ket(primitives))
                    else:
                        raise ValueError("Invalid primitives object.")
                else:
                    return self.state_preparation(Ket(primitives))
            case _:
                raise ValueError("Invalid primitives object.")