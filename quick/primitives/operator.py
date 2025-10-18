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
    Quantum operators are unitary matrices which represent operations applied to
    quantum states (represented with qubits). It uses LSB convention.

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
    `num_control_qubits` : int
        The number of qubits the operator uses as controls.
    `tensor_shape` : tuple[int, ...]
        The shape of the quantum operator tensor based on
        qubits as the physical dimension.

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
        self.num_control_qubits = 0
        self.tensor_shape = (2, 2) * self.num_qubits

    @classmethod
    def from_matrix(
            cls,
            matrix: NDArray[np.complex128]
        ) -> Operator:
        """ Create an `quick.primitives.Operator` from a matrix.

        Parameters
        ----------
        `matrix` : NDArray[np.complex128]
            The matrix to create the operator from. The matrix is not
            required to be unitary, but if it is not, we will approximate
            it to the nearest unitary matrix using Singular Value Decomposition (SVD).

        Returns
        -------
        quick.primitives.Operator
            The operator created from the matrix.
        """
        if is_unitary_matrix(matrix):
            return cls(matrix)

        U, _, Vh = np.linalg.svd(matrix)
        return cls(np.array(U @ Vh).astype(complex))

    def conj(self) -> Operator:
        """ Take the conjugate of the operator.

        Returns
        -------
        quick.primitives.Operator
            The conjugate of the operator.
        """
        return Operator(np.conjugate(self.data), label=self.label)

    def T(self) -> Operator:
        """ Take the transpose of the operator.

        Returns
        -------
        quick.primitives.Operator
            The transpose of the operator.
        """
        return Operator(np.transpose(self.data), label=self.label)

    def adjoint(self) -> Operator:
        """ Take the adjoint of the operator.

        Returns
        -------
        quick.primitives.Operator
            The adjoint of the operator.
        """
        return self.conj().T()

    def reverse_bits(self) -> None:
        """ Reverse the order of the qubits in the operator.
        This changes MSB to LSB, and vice versa.
        """
        axes = tuple(range(self.num_qubits - 1, -1, -1))
        axes = axes + tuple(len(axes) + i for i in axes)
        self.data = np.reshape(
            np.transpose(
                np.reshape(self.data, self.tensor_shape), axes
            ),
            self.shape
        )

    def contract(
            self,
            op: NDArray[np.complex128] | Operator,
            qubit_indices: list[int]
        ) -> None:
        """ Contract the operator with an operator on the specified
        qubits in place using Einstein's Summation convention.

        Notes
        -----
        This implementation should be used for small systems,
        as it requires significant memory and thus may not be
        suitable for larger systems.

        For larger systems, consider using the more efficient
        `quick.backend.QuimbBackend` which leverages optimal
        tensor network contraction for simulating the circuit.

        Alternatively, consider using GPU-based simulators
        present in `quick.backend` which can be faster at
        scale.

        Parameters
        ----------
        `op` : NDArray[np.complex128] | Operator
            The operator or matrix to contract with.
        `qubit_indices` : list[int]
            The qubit indices to contract over.

        Raises
        ------
        ValueError
            ValueError
            - If the operator is not unitary.
            - If the number of indices is less than the number of qubits for `op`.
            - If the number of qubit indices exceeds the number of qubits in `self`.
            - If any of the qubit indices are out of range of `self`.

        Usage
        -----
        >>> op1.contract(op2, [0, 1])
        """
        from quick.primitives.contraction import contract

        contract(self, op, qubit_indices)

    def control(
            self,
            num_controls: int = 1
        ) -> Operator:
        """ Generate the controlled version of the operator.

        Parameters
        ----------
        `num_controls` : int
            The number of control qubits.

        Returns
        -------
        quick.primitives.Operator
            The controlled version of the operator.

        Raises
        ------
        ValueError
            - If the number of control qubits is less than 1.
        """
        if num_controls < 1:
            raise ValueError(
                "Number of control qubits must be at least 1."
                f"Received {num_controls} instead."
            )

        self.num_control_qubits += num_controls

        zero_projector = np.array([
            [1, 0],
            [0, 0]
        ])
        one_projector = np.array([
            [0, 0],
            [0, 1]
        ])

        controlled_operator = self.data

        for _ in range(num_controls):
            control_component = np.kron(np.eye(controlled_operator.shape[0]), zero_projector).astype(np.complex128)
            target_component = np.kron(controlled_operator, one_projector).astype(np.complex128)
            controlled_operator = control_component + target_component

        return Operator(controlled_operator)

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

    def __array__(self) -> NDArray[np.complex128]:
        """ Convert the `quick.primitives.Operator` to a NumPy array.

        Returns
        -------
        NDArray[np.complex128]
        """
        return np.array(self.data).astype(np.complex128)

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
            other: Scalar
        ) -> Operator:
        ...

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
            other: Scalar | statevector.Statevector | Operator
        ) -> Operator | statevector.Statevector:
        """ Multiply an operator with a number, statevector, or another operator.

        Notes
        -----
        The multiplication of a number with the operator behaves like the global
        phase shift.

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
        if isinstance(other, Scalar):
            return Operator(self.data * complex(other))
        elif isinstance(other, statevector.Statevector):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply an operator with an incompatible statevector.")
            return statevector.Statevector((self.data @ other.data).astype(np.complex128))
        elif isinstance(other, Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible operators.")
            return Operator(self.data @ other.data)

        raise TypeError(f"Multiplication with {type(other)} is not supported.")

    def __matmul__(
            self,
            other: Operator
        ) -> Operator:
        """ Calculate the tensor product of the two operators.

        Parameters
        ----------
        `other` : quick.primitives.Operator
            The operator to tensor with.

        Returns
        -------
        quick.primitives.Operator
            The tensor product of the two operators.

        Raises
        ------
        TypeError
            - If the `other` is not a `quick.primitives.Operator` instance.
        """
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot tensor Operator with {type(other)}.")

        return Operator(np.kron(self.data, other.data))

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