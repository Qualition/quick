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

""" Abstract base class for defining state preparation methods
to prepare quantum states.
"""

from __future__ import annotations

import qickit.circuit.circuit

__all__ = ["StatePreparation"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Type, TYPE_CHECKING

import qickit
if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Bra, Ket


class StatePreparation(ABC):
    """ `qickit.synthesis.statepreparation.StatePreparation` is the class for preparing quantum states.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `qickit.circuit.Circuit`.

    Usage
    -----
    >>> state_preparer = StatePreparation(output_framework=QiskitCircuit)
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:
        """ Initalize a State Preparation instance.
        """
        if not issubclass(output_framework, qickit.circuit.circuit.Circuit):
            raise TypeError("The output framework must be a subclass of qickit.circuit.Circuit.")

        self.output_framework = output_framework

    def prepare_state(
            self,
            state: NDArray[np.complex128] | Bra | Ket,
            compression_percentage: float=0.0,
            index_type: Literal["row", "snake"]="row"
        ) -> Circuit:
        """ Prepare the quantum state.

        Parameters
        ----------
        `state` : NDArray[np.complex128] | qickit.primitives.Bra | qickit.primitives.Ket
            The quantum state to prepare.
        `compression_percentage` : float, optional, default=0.0
            Number between 0 an 100, where 0 is no compression and 100 all statevector values are 0.
        `index_type` : Literal["row", "snake"], optional, default="row"
            The indexing type for the data.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit that prepares the state.

        Raises
        ------
        TypeError
            - If the state is not a numpy array or a Bra/Ket object.
        """
        if not isinstance(state, (np.ndarray, Bra, Ket)):
            try:
                state = np.array(state).astype(complex)
            except (ValueError, TypeError):
                raise TypeError(f"The state must be a numpy array or a Bra/Ket object. Received {type(state)} instead.")

        if isinstance(state, np.ndarray):
            state = Ket(state)

        # Get the number of qubits needed to implement the state
        num_qubits = state.num_qubits

        # Initialize the qickit circuit
        circuit = self.output_framework(num_qubits)

        return self.apply_state(circuit, state, range(num_qubits), compression_percentage, index_type)

    @abstractmethod
    def apply_state(
            self,
            circuit: Circuit,
            state: NDArray[np.complex128] | Bra | Ket,
            qubit_indices: int | Sequence[int],
            compression_percentage: float=0.0,
            index_type: Literal["row", "snake"]="row"
        ) -> Circuit:
        """ Apply the quantum state to a quantum circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit to which the state is applied.
        `state` : NDArray[np.complex128] | qickit.primitives.Bra | qickit.primitives.Ket
            The quantum state to apply.
        `qubit_indices` : int | Sequence[int]
            The qubit indices to which the state is applied.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit with the state applied.

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
        """