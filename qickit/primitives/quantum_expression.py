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
from qickit.synthesis.statepreparation import StatePreparation, Mottonen
from qickit.synthesis.unitarypreparation import UnitaryPreparation, QiskitUnitaryTranspiler
from qickit.primitives import Bra, Ket, Operator

# <bra|ket> -> swap-test between |bra> and |ket> controlled by an ancillary qubit (being the overlap result)
# |ket><bra| -> measurement operator
# |ket>|ket>
# A|ket>
# <bra|A|ket> -> expectation value

class QuantumExpression:
    """ `qickit.primitives.QuantumExpression` is a class that represents a quantum expression.
    Quantum expressions are mathematical expressions comprised of quantum states and operators.
    This class enables the running of Bra-Ket operations on gate-based backends by synthesizing
    the overall expression to a quantum circuit and running on a specified backend.

    Notes
    -----
    Whilst Bra-Ket notation is a widely utilized abstraction of quantum mechanics, it is not directly
    supported by quantum computers, i.e., we cannot represent bras as quantum circuits. However, we can
    create protocols that emulate what theses operations would do, such as the swap-test protocol to calculate
    the inner product of a bra and ket.

    Additionally, certain operations are not unitary, i.e., the outer product which is also known as the
    measurement operator. This is not a reversible operation and cannot be represented as a quantum circuit.
    Instead, one may attempt to represent this operation as a mid-circuit measurement.

    Parameters
    ----------
    `expression` : list[Bra | Ket | Operator]
        The quantum expression to be evaluated. The expression is a list of quantum states and
        operators, in the order they should be applied.
    `circuit_framework` : type[qickit.circuit.Circuit]
        The circuit framework to use for the quantum expression.
    `backend` : qickit.backend.Backend
        The backend to run the quantum expression on.
    `state_preparation_method` : type[qickit.synthesis.StatePreparation], optional
        The state preparation method to use for preparing quantum states.
    `unitary_preparation_method` : type[qickit.synthesis.UnitaryPreparation], optional
        The unitary preparation method to use for preparing quantum operators.

    Attributes
    ----------
    `expression` : list[Bra | Ket | Operator]
        The quantum expression to be evaluated.
    `circuit_framework` : type[qickit.circuit.Circuit]
        The circuit framework to use for the quantum expression.
    `backend` : qickit.backend.Backend
        The backend to run the quantum expression on.
    `state_preparation_method` : type[qickit.synthesis.StatePreparation]
        The state preparation method to use for preparing quantum states.
    `unitary_preparation_method` : type[qickit.synthesis.UnitaryPreparation]
        The unitary preparation method to use for preparing quantum operators.

    Usage
    -----
    >>> expression = [bra, operator, ket]
    >>> quantum_expression = QuantumExpression(expression, backend)
    """
    def __init__(
            self,
            expression: list[Bra | Ket | Operator],
            circuit_framework: Type[Circuit],
            backend: Backend,
            state_preparation_method: Type[StatePreparation] | None = None,
            unitary_preparation_method: Type[UnitaryPreparation] | None = None
        ) -> None:
        """ Initialize a `QuantumExpression` instance.
        """
        self.check_expression(expression)
        self.circuit_framework = circuit_framework
        self.backend = backend

        if state_preparation_method is None:
            state_preparation_method = Mottonen
        self.state_preparation_method = state_preparation_method(circuit_framework)

        if unitary_preparation_method is None:
            unitary_preparation_method = QiskitUnitaryTranspiler
        self.unitary_preparation_method = unitary_preparation_method(circuit_framework)

    def check_expression(
            self,
            expression: list[Bra | Ket | Operator]
        ) -> None:
        """ Check if the quantum expression is valid.

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

    def innerproduct(
            self,
            bra: Bra,
            ket: Ket
        ) -> float:
        """ Calculate the inner product between a bra and a ket.

        innerproduct = <bra|ket>

        Notes
        -----
        The inner product of a bra and a ket is defined as <bra|ket>. This is calculated
        using the swap-test protocol, which is a quantum circuit that calculates the inner
        product of two quantum states.

        Parameters
        ----------
        `bra` : qickit.primitives.Bra
            The bra quantum state.
        `ket` : qickit.primitives.Ket
            The ket quantum state.

        Returns
        -------
        float
            The inner product of the bra and ket.

        Usage
        -----
        >>> inner_product = quantum_expression.innerproduct(bra, ket)
        """
        num_qubits = bra.num_qubits

        # Define the circuit used for performing the inner product.
        inner_product_circuit = self.circuit_framework(num_qubits+1)

        # Prepare the bra and key circuits and add them to the temporary circuit
        temp_circuit = self.circuit_framework(num_qubits)
        bra_circuit = self.state_preparation_method.prepare_state(bra) # type: ignore
        ket_circuit = self.state_preparation_method.prepare_state(ket) # type: ignore

        temp_circuit.add(bra_circuit, range(num_qubits))
        temp_circuit.add(ket_circuit, range(num_qubits))

        # Convert the temporary circuit to a controlled circuit (this is our SWAP test circuit)
        VU = temp_circuit.control(1)

        # Add the SWAP test circuit to the inner product circuit
        inner_product_circuit.H(0)
        inner_product_circuit.add(VU, range(num_qubits+1))
        inner_product_circuit.H(0)

        inner_product_circuit.measure(0)

        # Run the inner product circuit on the backend
        num_shots = 1000
        counts = self.backend.get_counts(inner_product_circuit, num_shots=num_shots)

        # Calculate the inner product value
        mean_val = (counts['0']-counts['1'])/num_shots

        return mean_val

    def evaluate(self) -> NDArray[np.complex128]:
        """ Evaluate the quantum expression and returns the result.

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

        match first_expression_term:
            case Bra() | Ket():
                circuit: Circuit = self.state_preparation_method.prepare_state(first_expression_term)
            case Operator():
                circuit: Circuit = self.unitary_preparation_method.prepare_unitary(first_expression_term) # type: ignore

        for expr in self.expression:
            match expr:
                case Bra() | Ket():
                    braket_circuit: Circuit = self.state_preparation_method.prepare_state(expr) # type: ignore
                    circuit.add(braket_circuit, qubit_indices)
                case Operator():
                    operator_circuit: Circuit = self.unitary_preparation_method.prepare_unitary(expr)
                    circuit.add(operator_circuit, qubit_indices)

        statevector = self.backend.get_statevector(circuit)
        return statevector