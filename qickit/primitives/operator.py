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

""" Operator matrix class for representing quantum unitary operators.
"""

from __future__ import annotations

__all__ = ["Operator"]

import numpy as np
from numpy.typing import NDArray
from typing import Any, overload, SupportsFloat, TypeAlias

from qickit.predicates import is_square_matrix, is_unitary_matrix
import qickit.primitives.ket as ket

# `Scalar` is a type alias that represents a scalar value that can be either
# a real number or a complex number.
Scalar: TypeAlias = SupportsFloat | complex


class Operator:
    """ `qickit.primitives.Operator` class is used to represent a quantum operator. Quantum operators
    are hermitian matrices (square, unitary matrices) which represent operations applied to quantum
    states (represented with qubits).

    Parameters
    ----------
    `data` : NDArray[np.complex128]
        The quantum operator data. If the data is not a complex type, it will be converted to complex.
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
        - If the operator is not a square matrix.
        - If the operator dimension is not a power of 2.
        - If the operator cannot be converted to complex type.
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
        """ Initialize a `qickit.primitives.Operator` instance.
        """
        if label is None:
            self.label = "\N{LATIN CAPITAL LETTER A}\N{COMBINING CIRCUMFLEX ACCENT}"
        self.is_unitary(data)
        self.data = data
        self.shape = self.data.shape
        self.num_qubits = int(np.ceil(np.log2(self.shape[0])))

    @staticmethod
    def is_unitary(data: NDArray[np.complex128]) -> None:
        """ Check if a matrix is Hermitian.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The matrix to check.

        Raises
        ------
        ValueError
            - If the matrix is not square.
            - If the matrix dimension is not a power of 2.
            - If the matrix cannot be converted to complex type.
            - If the matrix is not unitary.

        Usage
        -----
        >>> data = np.array([[1+0j, 0+0j],
        ...                  [0+0j, 1+0j]])
        >>> ishermitian(data)
        """
        # Check if the matrix is square
        if not is_square_matrix(data):
            raise ValueError("Operator must be a square matrix.")

        # Check if the matrix dimension is a power of 2
        if not ((data.shape[0] & (data.shape[0]-1) == 0) and data.shape[0] != 0):
            raise ValueError("Operator dimension must be a power of 2.")

        # Check if the data type is complex
        if not np.iscomplexobj(data):
            try:
                data = data.astype(np.complex128)
            except ValueError:
                raise ValueError("Cannot convert data to complex type.")

        # Check if the matrix is unitary
        if not is_unitary_matrix(data):
            raise ValueError("Operator must be unitary.")

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
            - If the the operator and ket are incompatible.
            - If the two operators are incompatible.
        NotImplementedError
            - If the `other` type is incompatible.
        """
        if isinstance(other, (SupportsFloat, complex)):
            return
        elif isinstance(other, ket.Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply an operator with an incompatible ket.")
        elif isinstance(other, Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible operators.")
        else:
            raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    @overload
    def __mul__(
            self,
            other: Scalar
        ) -> Operator:
        ...

    @overload
    def __mul__(
            self,
            other: ket.Ket
        ) -> ket.Ket:
        ...

    @overload
    def __mul__(
            self,
            other: Operator
        ) -> Operator:
        ...

    def __mul__(
            self,
            other: Scalar | ket.Ket | Operator
        ) -> Operator | ket.Ket:
        """ Multiply an operator with a scalar, ket or another operator.

        The multiplication of an operator with a ket is defined as:
        - A|ψ⟩ = |ψ'⟩

        The multiplication of an operator with another operator is defined as:
        - AB = C

        Notes
        -----
        The multiplication of an operator with a scalar does not change the operator. This is because
        the norm of the operator is preserved, and the scalar is multiplied with each element of the
        operator. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : qickit.primitives.Scalar | qickit.primitives.Ket | qickit.primitives.Operator
            The object to multiply with.

        Returns
        -------
        qickit.primitives.Operator | qickit.primitives.Ket
            The result of the multiplication.

        Raises
        ------
        ValueError
            - If the operator and ket dimensions are incompatible.
            - If the operator dimensions are incompatible.
        NotImplementedError
            - If the `other` type is incompatible.

        Usage
        -----
        >>> scalar = 2
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> operator * scalar
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> ket = Ket([1+0j, 0+0j])
        >>> operator * ket
        >>> operator1 = Operator([[1+0j, 0+0j],
        ...                       [0+0j, 1+0j]])
        >>> operator2 = Operator([[1+0j, 0+0j],
        ...                       [0+0j, 1+0j]])
        >>> operator1 * operator2
        """
        if isinstance(other, (SupportsFloat, complex)):
            return Operator((self.data * other).astype(np.complex128)) # type: ignore
        elif isinstance(other, ket.Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply an operator with an incompatible ket.")
            return ket.Ket((self.data @ other.data).astype(np.complex128)) # type: ignore
        elif isinstance(other, Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible operators.")
            return Operator(self.data @ other.data)
        else:
            raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __rmul__(
            self,
            other: Scalar
        ) -> Operator:
        """ Multiply a scalar with an operator.

        Notes
        -----
        The multiplication of an operator with a scalar does not change the operator. This is because
        the norm of the operator is preserved, and the scalar is multiplied with each element of the
        operator. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : qickit.primitives.Scalar
            The scalar to multiply with.

        Returns
        -------
        qickit.primitives.Operator
            The operator multiplied by the scalar.

        Raises
        ------
        NotImplementedError
            - If the `other` type is incompatible

        Usage
        -----
        >>> scalar = 2
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> scalar * operator
        """
        if isinstance(other, (SupportsFloat, complex)):
            return Operator((self.data * other).astype(np.complex128)) # type: ignore

        raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __str__(self) -> str:
        """ Return the string representation of the operator.

        Usage
        -----
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> print(operator)
        """
        return f"{self.label}"

    def __repr__(self) -> str:
        """ Return the string representation of the operator.

        Usage
        -----
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> repr(operator)
        """
        return f"Operator(data={self.data})"