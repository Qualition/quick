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

""" Abstract base class for defining unitary preparation methods
to prepare quantum operators.
"""

from __future__ import annotations

__all__ = ["UnitaryPreparation"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Type, TYPE_CHECKING

import qickit
if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Operator


class UnitaryPreparation(ABC):
    """ `qickit.UnitaryPreparation` is the class for preparing quantum operators.

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
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:
        """ Initalize a Unitary Preparation instance.
        """
        if not issubclass(output_framework, qickit.circuit.Circuit):
            raise TypeError("The output framework must be a subclass of qickit.circuit.Circuit.")

        self.output_framework = output_framework

    def prepare_unitary(
            self,
            unitary: NDArray[np.complex128] | Operator
        ) -> Circuit:
        """ Prepare the quantum unitary operator.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The quantum unitary operator.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit for preparing the unitary operator.

        Usage
        -----
        >>> unitary_preparation.prepare_unitary(unitary)
        """
        if not isinstance(unitary, (np.ndarray, Operator)):
            try:
                unitary = np.array(unitary).astype(complex)
            except (ValueError, TypeError):
                raise TypeError(f"The operator must be a numpy array or an Operator object. Received {type(unitary)} instead.")

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        # Get the number of qubits needed to implement the operator
        num_qubits = unitary.num_qubits

        # Initialize the qickit circuit
        circuit = self.output_framework(num_qubits)

        # Apply the unitary matrix to the circuit
        # and return the circuit
        return self.apply_unitary(circuit, unitary, range(num_qubits))

    @abstractmethod
    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:
        """ Apply the quantum unitary operator to a quantum circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The quantum unitary operator.
        `qubit_indices` : int | Sequence[int]
            The qubit indices to apply the unitary operator to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit with the unitary operator applied.

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
        >>> unitary_preparation.apply_unitary(circuit, unitary, qubit_indices)
        """