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

__all__ = ["QuantumExpression"]

from itertools import pairwise
import numpy as np
from numpy.typing import NDArray
from typing import Type

from qickit.circuit import Circuit
from qickit.backend import Backend
from qickit.synthesis.statepreparation import StatePreparation
from qickit.synthesis.unitarypreparation import UnitaryPreparation
from qickit.primitives import Bra, Ket, Operator


class QuantumExpression:
    """ `qickit.primitives.QuantumExpression` is a class that represents a quantum expression.
    Quantum expressions are mathematical expressions comprised of quantum states and operators.
    This class enables the running of Bra-Ket operations on gate-based backends by synthesizing
    the overall expression to a quantum circuit and running on a specified backend.

    Parameters
    ----------
    `expression` : list[Bra | Ket | Operator]
        The quantum expression to be evaluated. The expression is a list of quantum states and
        operators, in the order they should be applied.
    `backend` : qickit.backend.Backend
        The backend to run the quantum expression on.

    Attributes
    ----------
    `expression` : list[Bra | Ket | Operator]
        The quantum expression to be evaluated.
    `backend` : qickit.backend.Backend
        The backend to run the quantum expression on.

    Usage
    -----
    >>> expression = [bra, operator, ket]
    >>> quantum_expression = QuantumExpression(expression, backend)
    """
    def __init__(self,
                 expression: list[Bra | Ket | Operator],
                 backend: Backend,
                 state_preparation_method: Type[StatePreparation] | None = None,
                 unitary_preparation_method: Type[UnitaryPreparation] | None = None) -> None:
        """ Initializes a `QuantumExpression` instance.
        """
        self.check_expression(expression)
        self.backend = backend
        self.state_preparation_method = state_preparation_method
        self.unitary_preparation_method = unitary_preparation_method

    def check_expression(self,
                         expression: list[Bra | Ket | Operator]) -> None:
        """ Checks if the quantum expression is valid.

        Parameters
        ----------
        `expression` : list[Bra | Ket | Operator]
            The quantum expression to be evaluated.

        Raises
        ------
        ValueError
            If the expression is empty.
            If the expression contains an invalid element.

        Usage
        -----
        >>> expression = [ket, operator, ket]
        >>> quantum_expression.check_expression(expression)
        """
        if len(expression) == 0:
            raise ValueError("Expression cannot be empty.")

        for a, b in pairwise(expression):
            if not isinstance(a, (Bra, Ket, Operator)) or not isinstance(b, (Bra, Ket, Operator)):
                raise ValueError("Expression must contain only Bra, Ket, or Operator objects.")
            a._check__mul__(b)

        self.expression = expression

    def evaluate(self) -> NDArray[np.complex128]:
        """ Evaluates the quantum expression and returns the result.

        Returns
        -------
        NDArray[np.complex128]
            The resulting state of the quantum expression.

        Usage
        -----
        >>> expression.evaluate()
        """
        qubit_indices = list(range(self.expression[0].num_qubits))
        first_expression_term = self.expression[0]

        if isinstance(first_expression_term, (Bra, Ket)):
            circuit: Circuit = self.state_preparation_method.prepare_state(first_expression_term)
        elif isinstance(first_expression_term, Operator):
            circuit: Circuit = self.unitary_preparation_method.prepare_unitary(first_expression_term)

        for expr in self.expression:
            if isinstance(expr, (Bra, Ket)):
                braket_circuit: Circuit = self.state_preparation_method.prepare_state(expr)
                circuit.add(braket_circuit, qubit_indices)
            elif isinstance(expr, Operator):
                operator_circuit: Circuit = self.unitary_preparation_method.prepare_unitary(expr)
                circuit.add(operator_circuit, qubit_indices)

        statevector = self.backend.get_statevector(circuit)
        return statevector