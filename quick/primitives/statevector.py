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

""" Statevector class for representing quantum (Ket) state vectors.
"""

from __future__ import annotations

__all__ = ["Statevector"]

import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal, SupportsFloat, TypeAlias

import quick.primitives.operator as operator

# `Scalar` is a type alias that represents a scalar value that can be either
# a real number or a complex number.
Scalar: TypeAlias = SupportsFloat | complex


class Statevector:
    """ `quick.primitives.Statevector` is a class that represents a qubit
    statevector. Qubit statevectors are complex vectors with a magnitude
    of 1 with 2^N elements where N is the number of qubits used to represent
    the statevector.

    Parameters
    ----------
    `data` : NDArray[np.complex128]
        The statevector data. The data will be normalized
        to 2-norm and padded if necessary.
    `label` : str, optional
        The label of the statevector.

    Attributes
    ----------
    `label` : str, optional, default="Ψ"
        The label of the statevector.
    `data` : NDArray[np.complex128]
        The statevector data.
    `norm_scale` : np.float64
        The normalization scale.
    `normalized` : bool
        Whether the statevector is normalized to 2-norm or not.
    `num_qubits` : int
        The number of qubits represented by the statevector.
    `tensor_shape` : tuple[int, ...]
        The shape of the statevector tensor based on qubits
        as the physical dimension.

    Raises
    ------
    ValueError
        - If the data is a scalar or an operator.

    Usage
    -----
    >>> data = np.array([1, 2, 3, 4])
    >>> statevector = Statevector(data)
    """
    def __init__(
            self,
            data: NDArray[np.complex128],
            label: str | None = None
        ) -> None:
        """ Initialize a `quick.primitives.Statevector` instance.
        """
        if label is None:
            self.label = "\N{GREEK CAPITAL LETTER PSI}"
        else:
            self.label = label

        data = np.array(data)
        self.validate_data(data)
        self.data = data.flatten().astype(np.complex128)
        self.norm_scale = np.linalg.norm(self.data)
        self.num_qubits = int(np.ceil(np.log2(self.data.size)))
        self.tensor_shape = (2,) * self.num_qubits
        self.is_normalized()
        self.is_padded()
        self.to_quantumstate()

    @staticmethod
    def validate_data(data: NDArray[np.complex128]) -> None:
        """ Validate the data to ensure it is a valid statevector.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data to validate.

        Raises
        ------
        ValueError
            - If the data is a scalar or an operator.
        """
        if isinstance(data, Scalar) and data.size == 1:
            raise ValueError("Cannot convert a scalar to a statevector.")
        elif data.ndim == 0 or data.size == 1:
            raise ValueError("Cannot convert a scalar to a statevector.")
        elif data.ndim == 2 and data.shape[0] != 1:
            raise ValueError("Cannot convert an operator to a statevector.")

    @staticmethod
    def check_normalization(data: NDArray[np.complex128]) -> bool:
        """ Check if a data is normalized to 2-norm.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data.

        Returns
        -------
        bool
            Whether the vector is normalized to 2-norm or not.

        Usage
        -----
        >>> data = np.array([1, 2, 3, 4])
        >>> check_normalization(data)
        """
        return bool(
            np.isclose(
                np.linalg.norm(data), 1.0, atol=1e-08
            )
        )

    def is_normalized(self) -> None:
        """ Check if a `quick.primitives.Statevector` instance is normalized to 2-norm.

        Usage
        -----
        >>> statevector.is_normalized()
        """
        self.normalized = self.check_normalization(self.data)

    @staticmethod
    def normalize_data(
            data: NDArray[np.complex128],
            norm_scale: np.float64
        ) -> NDArray[np.complex128]:
        """ Normalize the data to 2-norm, and return the normalized data.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data.
        `norm_scale` : np.float64
            The normalization scale.

        Returns
        -------
        NDArray[np.complex128]
            The 2-norm normalized data.

        Usage
        -----
        >>> data = np.array([[1, 2],
        ...                  [3, 4]])
        >>> norm_scale = np.linalg.norm(data.flatten())
        >>> normalize_data(data, norm_scale)
        """
        return np.multiply(data, 1/norm_scale)

    def normalize(self) -> None:
        """ Normalize a `quick.primitives.Statevector` instance to 2-norm.

        Usage
        -----
        >>> statevector.normalize()
        """
        if self.normalized:
            return

        self.data = self.normalize_data(self.data, self.norm_scale)
        self.normalized = True

    @staticmethod
    def check_padding(data: NDArray[np.complex128]) -> bool:
        """ Check if a data is normalized to 2-norm.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data.

        Returns
        -------
        bool
            Whether the vector is normalized to 2-norm or not.

        Usage
        -----
        >>> data = np.array([[1, 2], [3, 4]])
        >>> check_padding(data)
        """
        len_data = len(data)
        return (len_data & (len_data - 1) == 0) and len_data != 0

    def is_padded(self) -> None:
        """ Check if a `quick.primitives.Statevector` instance is padded to a power of 2.

        Usage
        -----
        >>> statevector.is_padded()
        """
        self.padded = self.check_padding(self.data)

    @staticmethod
    def pad_data(
            data: NDArray[np.complex128],
            target_size: int
        ) -> NDArray[np.complex128]:
        """ Pad data with zeros up to the nearest power of 2, and return
        the padded data.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data to be padded.
        `target_size` : int
            The target size to pad the data to.

        Returns
        -------
        `padded_data` : NDArray[np.complex128]
            The padded data.

        Usage
        -----
        >>> data = np.array([1, 2, 3])
        >>> pad_data(data, 4)
        """
        padded_data = np.pad(
            data, (0, int(target_size - len(data))),
            mode="constant"
        )

        return padded_data

    def pad(self) -> None:
        """ Pad a `quick.primitives.Statevector` instance.

        Usage
        -----
        >>> statevector.pad()
        """
        if self.padded:
            return

        self.data = self.pad_data(self.data, 2 ** self.num_qubits)
        self.padded = True

    def to_quantumstate(self) -> None:
        """ Ensure the statevector is in a valid quantum state.

        Usage
        -----
        >>> statevector.to_quantumstate()
        """
        if not self.normalized:
            self.normalize()

        if not self.padded:
            self.pad()

    def compress(
            self,
            compression_percentage: float
        ) -> None:
        """ Compress a `quick.primitives.Statevector` instance.

        Parameters
        ----------
        `compression_percentage` : float
            The percentage of compression.

        Usage
        -----
        >>> statevector.compress(50)
        """
        data_sort_ind = np.argsort(np.abs(self.data))

        # Set the smallest absolute values of data to zero according to compression parameter
        cutoff = int((compression_percentage / 100.0) * len(self.data))
        for i in data_sort_ind[:cutoff]:
            self.data[i] = 0

    def change_indexing(
            self,
            index_type: Literal["row", "snake"]
        ) -> None:
        """ Change the indexing of a `quick.primitives.Statevector` instance.

        Parameters
        ----------
        `index_type` : Literal["row", "snake"]
            The new indexing type, being "row" or "snake".

        Raises
        ------
        ValueError
            - If the index type is not supported.

        Usage
        -----
        >>> statevector.change_indexing("snake")
        """
        if index_type == "snake":
            if self.num_qubits >= 3:
                # Convert the statevector to a matrix (image)
                self.data = self.data.reshape(2, -1)
                # Reverse the elements in odd rows
                self.data[1::2, :] = self.data[1::2, ::-1]

                self.data = self.data.flatten()
        elif index_type == "row":
            self.data = self.data
        else:
            raise ValueError("Index type not supported.")

    def reverse_bits(self) -> None:
        """ Reverse the order of the qubits in the statevector.
        This changes MSB to LSB, and vice versa.
        """
        self.data = np.transpose(
            np.reshape(
                self.data,
                self.tensor_shape
            )
        ).ravel()

    def trace(self) -> float:
        """ Calculate the trace of the statevector.

        Returns
        -------
        float
            The trace of the statevector.

        Usage
        -----
        >>> statevector.trace()
        """
        return float(np.sum(np.abs(self.data) ** 2))

    def partial_trace(
            self,
            trace_qubit_indices: list[int]
        ) -> float | NDArray[np.complex128]:
        """ Calculate the partial trace of the statevector.

        Parameters
        ----------
        `trace_qubit_indices` : list[int]
            The indices of the qubits to trace out.

        Returns
        -------
        `rho` : float | NDArray[np.complex128]
            The resulting density matrix after tracing out the specified qubits.
            If the `trace_qubit_indices` match the total number of qubits, then
            we return the trace of the statevector.

        Raises
        ------
        ValueError
            - If the trace qubit indices are invalid.

        Usage
        -----
        >>> statevector.partial_trace([0, 1])
        """
        for i in trace_qubit_indices:
            if not 0 <= i < self.num_qubits:
                raise ValueError(
                    f"Invalid trace qubit index {i}. "
                    f"Valid indices are in range(0, {self.num_qubits})."
                )

        num_traced_qubits = len(trace_qubit_indices)

        if num_traced_qubits == self.num_qubits:
            return self.trace()

        traced_shape = (2**(self.num_qubits - num_traced_qubits),) * 2
        trace_systems = [self.num_qubits - 1 - i for i in trace_qubit_indices]
        state = self.data.reshape(self.tensor_shape)
        rho = np.tensordot(state, state.conj(), axes=(trace_systems, trace_systems))
        rho = np.reshape(rho, traced_shape)

        return rho

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
            - If the two vectors are incompatible.
            - If the the statevector and operator are incompatible.
        TypeError
            - If the `other` type is incompatible.
        """
        if isinstance(other, (SupportsFloat, complex)):
            return
        elif isinstance(other, operator.Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible vectors.")
        else:
            raise TypeError(f"Multiplication with {type(other)} is not supported.")

    def __eq__(
            self,
            other: object
        ) -> bool:
        """ Check if two statevector vectors are equal.

        Parameters
        ----------
        `other` : object
            The other statevector vector.

        Returns
        -------
        bool
            Whether the two statevector vectors are equal.

        Raises
        ------
        TypeError
            - If the `other` object is not a `quick.primitives.Statevector` instance.

        Usage
        -----
        >>> statevector1 = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector2 = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector1 == statevector2
        """
        if not isinstance(other, Statevector):
            raise TypeError(
                "Statevector can only be compared with other Statevector instances. "
                f"Received {type(other)} instead."
            )

        return bool(np.all(np.isclose(self.data, other.data, atol=1e-8, rtol=0)))

    def __len__(self) -> int:
        """ Return the length of the statevector vector.

        Returns
        -------
        int
            The length of the statevector vector.

        Usage
        -----
        >>> len(statevector)
        """
        return len(self.data)

    def __add__(
            self,
            other: Statevector
        ) -> Statevector:
        """ Superpose two statevector states together.

        Parameters
        ----------
        `other` : quick.primitives.Statevector
            The other statevector state.

        Returns
        -------
        quick.primitives.Statevector
            The superposed statevector state.

        Raises
        ------
        TypeError
            - If the `other` object is not a `quick.primitives.Statevector` instance.
        ValueError
            - If the two statevectors are incompatible.

        Usage
        -----
        >>> statevector1 = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector2 = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector1 + statevector2
        """
        if not isinstance(other, Statevector):
            raise TypeError(
                "Statevector can only be added to other Statevector instances. "
                f"Received {type(other)} instead."
            )

        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot add two incompatible vectors.")

        return Statevector((self.data + other.data).astype(np.complex128))

    def __mul__(
            self,
            other: Scalar
        ) -> Statevector:
        """ Multiply the statevector by a scalar.

        Notes
        -----
        The multiplication of a statevector with a scalar does not change
        the statevector. This is because the distribution of the statevector
        is preserved as the scalar is multiplied with each element of the
        statevector. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : Scalar
            The other object to multiply the statevector by.

        Returns
        -------
        quick.primitives.Statevector
            The result of the multiplication.

        Raises
        ------
        TypeError
            - If the `other` object is not a number.

        Usage
        -----
        >>> scalar = 2
        >>> statevector = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector * scalar
        """
        if not isinstance(other, Scalar):
            raise TypeError(
                "Statevector can only be multiplied by a scalar. "
                f"Received {type(other)} instead."
            )

        return Statevector(
            (self.data * complex(other)).astype(np.complex128)
        )

    def __rmul__(
            self,
            other: Scalar
        ) -> Statevector:
        """ Multiply the statevector by a scalar.

        Notes
        -----
        The multiplication of a statevector with a scalar does not change
        the statevector. This is because the distribution of the statevector
        is preserved as the scalar is multiplied with each element of the
        statevector. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : Scalar
            The scalar to multiply the statevector by.

        Returns
        -------
        quick.primitives.Statevector
            The statevector multiplied by the scalar.

        Raises
        ------
        TypeError
            - If the `other` object is not a number.

        Usage
        -----
        >>> scalar = 2
        >>> statevector = Statevector(np.array([1+0j, 0+0j]))
        >>> scalar * statevector
        """
        if not isinstance(other, Scalar):
            raise TypeError(
                "Statevector can only be multiplied by a scalar. "
                f"Received {type(other)} instead."
            )

        return Statevector(
            (self.data * complex(other)).astype(np.complex128)
        )

    def __matmul__(
            self,
            other: Statevector
        ) -> Statevector:
        """ Tensor product of two statevectors.

        Parameters
        ----------
        `other` : quick.primitives.Statevector
            The statevector to tensor product with.

        Returns
        -------
        quick.primitives.Statevector
            The resulting statevector.

        Raises
        ------
        TypeError
            - If `other` is not a `quick.primitives.Statevector`.

        Usage
        -----
        >>> statevector1 = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector2 = Statevector(np.array([1+0j, 0+0j]))
        >>> statevector1 @ statevector2
        """
        if not isinstance(other, Statevector):
            raise TypeError(
                "Statevector can only be tensored with other Statevector instances. "
                f"Received {type(other)} instead."
            )

        return Statevector(
            np.kron(self.data, other.data).astype(np.complex128)
        )

    def __str__(self) -> str:
        """ Return the string representation of the statevector vector.

        Returns
        -------
        str
            The string representation of the statevector vector.

        Usage
        -----
        >>> str(statevector)
        """
        return f"|{self.label}⟩"

    def __repr__(self) -> str:
        """ Return the string representation of the statevector vector.

        Returns
        -------
        str
            The string representation of the statevector vector.

        Usage
        -----
        >>> repr(statevector)
        """
        return f"{self.__class__.__name__}(data={self.data}, label={self.label})"