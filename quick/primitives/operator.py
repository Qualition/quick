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

""" Operator matrix class for representing quantum unitary operators.
"""

from __future__ import annotations

__all__ = ["Operator"]

import numpy as np
from numpy.typing import NDArray
from typing import Any, overload, SupportsFloat, TypeAlias

from quick.predicates import is_unitary_matrix
import quick.primitives.statevector as statevector

# `Scalar` is a type alias that represents a scalar value that can be either
# a real number or a complex number.
Scalar: TypeAlias = SupportsFloat | complex


class Operator:
    """ `quick.primitives.Operator` class is used to represent a quantum operator.
    Quantum operators are hermitian matrices (square, unitary matrices) which represent
    operations applied to quantum states (represented with qubits).

    Parameters
    ----------
    `data` : NDArray[np.complex128]
        The quantum operator data. If the data is not a complex type,
        it will be converted to complex.
    `label` : str, optional
        The label of the quantum operator.

    Attributes
    ----------
    `label` : str, optional, default="A"
        The label of the quantum operator.
    `data` : NDArray[np.complex128]
        The quantum operator data.
    `shape` : tuple[int, int]
        The shape of the quantum operator data.
    `num_qubits` : int
        The number of qubits the quantum operator acts on.

    Raises
    ------
    ValueError
        - If the operator is not unitary.

    Usage
    -----
    >>> data = np.array([[1+0j, 0+0j],
    ...                  [0+0j, 1+0j]])
    >>> Operator(data)
    """
    def __init__(
            self,
            data: NDArray[np.complex128],
            label: str | None = None
        ) -> None:
        """ Initialize a `quick.primitives.Operator` instance.
        """
        if label is None:
            self.label = "\N{LATIN CAPITAL LETTER A}\N{COMBINING CIRCUMFLEX ACCENT}"
        else:
            self.label = label

        data = np.array(data)

        if not is_unitary_matrix(data):
            raise ValueError("Operator must be unitary.")

        self.data = data
        self.shape = self.data.shape
        self.num_qubits = int(np.ceil(np.log2(self.shape[0])))

    def _check__mul__(
            self,
            other: Any
        ) -> None:
        """ Check if the multiplication is valid.

        Parameters
        ----------
        `other` : Any
            The other object to multiply with.

        Raises
        ------
        ValueError
            - If the the operator and statevector are incompatible.
            - If the two operators are incompatible.
        TypeError
            - If the `other` type is incompatible.
        """
        if isinstance(other, statevector.Statevector):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply an operator with an incompatible statevector.")
        elif isinstance(other, Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible operators.")
        else:
            raise TypeError(f"Multiplication with {type(other)} is not supported.")

    def __eq__(
            self,
            other: Any
        ) -> bool:
        """ Check if two operators are equal.

        Parameters
        ----------
        `other` : Any
            The other object to compare with.

        Returns
        -------
        bool
            True if the operators are equal, False otherwise.

        Raises
        ------
        TypeError
            - If the `other` type is incompatible.
        """
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}.")

        return bool(np.all(np.isclose(self.data, other.data, atol=1e-8, rtol=0)))

    @overload
    def __mul__(
            self,
            other: statevector.Statevector
        ) -> statevector.Statevector:
        ...

    @overload
    def __mul__(
            self,
            other: Operator
        ) -> Operator:
        ...

    def __mul__(
            self,
            other: statevector.Statevector | Operator
        ) -> Operator | statevector.Statevector:
        """ Multiply an operator with a statevector or another operator.

        The multiplication of an operator with a statevector is defined as:
        - A|ψ⟩ = |ψ'⟩

        The multiplication of an operator with another operator is defined as:
        - AB = C

        Parameters
        ----------
        `other` : quick.primitives.Statevector | quick.primitives.Operator
            The object to multiply with.

        Returns
        -------
        quick.primitives.Operator | quick.primitives.Statevector
            The result of the multiplication.

        Raises
        ------
        ValueError
            - If the operator and ket dimensions are incompatible.
            - If the operator dimensions are incompatible.
        TypeError
            - If the `other` type is incompatible.

        Usage
        -----
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> statevector = Statevector([1+0j, 0+0j])
        >>> operator * statevector
        >>> operator1 = Operator([[1+0j, 0+0j],
        ...                       [0+0j, 1+0j]])
        >>> operator2 = Operator([[1+0j, 0+0j],
        ...                       [0+0j, 1+0j]])
        >>> operator1 * operator2
        """
        if isinstance(other, statevector.Statevector):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply an operator with an incompatible statevector.")
            return statevector.Statevector((self.data @ other.data).astype(np.complex128))
        elif isinstance(other, Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible operators.")
            return Operator(self.data @ other.data)

        raise TypeError(f"Multiplication with {type(other)} is not supported.")

    def __str__(self) -> str:
        """ Return the string representation of the operator.

        Usage
        -----
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> print(operator)
        """
        return self.label

    def __repr__(self) -> str:
        """ Return the string representation of the operator.

        Usage
        -----
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> repr(operator)
        """
        return f"Operator(data={self.data}, label={self.label})"