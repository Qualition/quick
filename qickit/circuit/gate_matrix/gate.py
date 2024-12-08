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

""" Module for generating the matrix representation of a quantum gate.
"""

from __future__ import annotations

__all__ = ["Gate"]

import numpy as np
from numpy.typing import NDArray
from typing import Literal

from qickit.predicates import is_unitary_matrix


class Gate:
    """ `qickit.gate_matrix.Gate` class represents a quantum gate. This class is used to
    generate the matrix representation of a quantum gate for testing and classical simulation
    purposes.

    Parameters
    ----------
    `name`: str
        The name of the gate.
    `matrix`: NDArray[np.complex128]
        The matrix representation of the gate.

    Attributes
    ----------
    `name`: str
        The name of the gate.
    `matrix`: NDArray[np.complex128]
        The matrix representation of the gate.

    Raises
    ------
    ValueError
        - If the matrix is not unitary.

    Usage
    -----
    >>> gate = Gate("H", np.array([[1, 1],
    ...                            [1, -1]]) / np.sqrt(2))
    """
    def __init__(
            self,
            name: str,
            matrix: NDArray[np.complex128]
        ) -> None:
        """ Initialize a `qickit.gate_matrix.Gate` instance.
        """
        self.name = name
        self.matrix = matrix
        if not is_unitary_matrix(matrix):
            raise ValueError("The matrix must be unitary.")
        self.num_qubits = int(np.log2(matrix.shape[0]))
        self.ordering = "MSB"

    def adjoint(self) -> NDArray[np.complex128]:
        """ Generate the adjoint of the gate.

        Returns
        -------
        NDArray[np.complex128]
            The adjoint of the gate.
        """
        return self.matrix.T.conj()

    def control(
            self,
            num_control_qubits: int
        ) -> Gate:
        """ Generate the matrix representation of a controlled version of the gate.

        Parameters
        ----------
        `num_control_qubits`: int
            The number of control qubits.

        Returns
        -------
        `controlled_gate` : qickit.gate_matrix.Gate
            The controlled gate.

        Raises
        ------
        TypeError
            - If the number of control qubits is not an integer.
        ValueError
            - If the number of control qubits is less than 1.
        """
        if not isinstance(num_control_qubits, int):
            raise TypeError("The number of control qubits must be an integer.")

        if num_control_qubits < 1:
            raise ValueError("The number of control qubits must be greater than 0.")

        zero_projector = np.array([
            [1, 0],
            [0, 0]
        ])
        one_projector = np.array([
            [0, 0],
            [0, 1]
        ])

        controlled_matrix = np.kron(zero_projector, np.eye(2 ** num_control_qubits)) + \
                            np.kron(one_projector, self.matrix)
        controlled_gate = Gate(f"C-{self.name}", controlled_matrix.astype(complex))

        return controlled_gate

    def change_mapping(
            self,
            ordering: Literal["MSB", "LSB"]
        ) -> None:
        """ Change the mapping of the qubits in the matrix representation of the gate.

        Parameters
        ----------
        `ordering`: Literal["MSB", "LSB"]
            The new qubit ordering.

        Returns
        -------
        `reordered_matrix` : NDArray[np.complex128]
            The new matrix with LSB conversion.

        Raises
        ------
        ValueError
            - If the ordering is not "MSB" or "LSB".
        """
        if ordering not in ["MSB", "LSB"]:
            raise ValueError("The ordering must be either 'MSB' or 'LSB'.")

        if ordering == self.ordering:
            return

        # Create a new matrix to store the reordered elements
        dims = [2] * (self.num_qubits * 2)
        reordered_matrix = self.matrix.reshape(dims).transpose().reshape(
            (2**self.num_qubits, 2**self.num_qubits)
        )

        self.matrix = reordered_matrix
        self.ordering = ordering