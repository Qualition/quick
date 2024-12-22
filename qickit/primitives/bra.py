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

""" Bra vector class for representing bra states.
"""

from __future__ import annotations

__all__ = ["Bra"]

import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal, overload, SupportsFloat, TypeAlias

import qickit.primitives.operator as operator
import qickit.primitives.ket as ket

# `Scalar` is a type alias that represents a scalar value that can be either
# a real number or a complex number.
Scalar: TypeAlias = SupportsFloat | complex


class Bra:
    """ `qickit.primitives.Bra` is a class that represents a quantum bra vector. Bra vectors are
    complex, row vectors with a magnitude of 1 which represent quantum states. The bra vectors are
    the complex conjugates of the ket vectors.

    Parameters
    ----------
    `data` : NDArray[np.complex128]
        The bra vector data. The data will be normalized to 2-norm and padded if necessary.
    `label` : str, optional
        The label of the bra vector.

    Attributes
    ----------
    `label` : str, optional, default="Ψ"
        The label of the bra vector.
    `data` : NDArray[np.complex128]
        The bra vector data.
    `norm_scale` : np.float64
        The normalization scale.
    `normalized` : bool
        Whether the bra vector is normalized to 2-norm or not.
    `shape` : Tuple[int, int]
        The shape of the bra vector.
    `num_qubits` : int
        The number of qubits represented by the bra vector.

    Raises
    ------
    ValueError
        - If the data is a scalar or an operator.

    Usage
    -----
    >>> data = np.array([1, 2, 3, 4])
    >>> bra = Bra(data)
    """
    def __init__(
            self,
            data: NDArray[np.complex128],
            label: str | None = None
        ) -> None:
        """ Initialize a `qickit.primitives.Bra` instance.
        """
        if label is None:
            self.label = "\N{GREEK CAPITAL LETTER PSI}"
        else:
            self.label = label

        self.norm_scale = np.linalg.norm(data)
        self.data = data
        self.shape = data.shape
        self.num_qubits = int(np.ceil(np.log2(len(data.flatten()))))
        self.is_normalized()
        self.is_padded()
        self.to_bra(data)

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
        # Check whether the data is normalized to 2-norm
        sum_check = np.sum(np.power(data, 2))

        # Check if the sum of squared of the data elements is equal to
        # 1 with 1e-8 tolerance
        return bool(np.isclose(sum_check, 1.0, atol=1e-08))

    def is_normalized(self) -> None:
        """ Check if a `qickit.primitives.Bra` instance is normalized to 2-norm.

        Usage
        -----
        >>> data.is_normalized()
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
        """ Normalize a `qickit.primitives.Bra` instance to 2-norm.
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
        return (data.shape[0] & (data.shape[0]-1) == 0) and data.shape[0] != 0

    def is_padded(self) -> None:
        """ Check if a `qickit.data.Data` instance is padded to a power of 2.

        Usage
        -----
        >>> data.is_padded()
        """
        self.padded = self.check_padding(self.data)

    @staticmethod
    def pad_data(
            data: NDArray[np.complex128],
            target_size: int
        ) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
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
        `data_shape` : (tuple[int, ...])
            The updated shape.

        Usage
        -----
        >>> data = np.array([1, 2, 3])
        >>> pad_data(data, 4)
        """
        padded_data = np.pad(
            data, (0, int(target_size - len(data))),
            mode="constant"
        )
        updated_shape = padded_data.shape

        return padded_data, updated_shape

    def pad(self) -> None:
        """ Pad a `qickit.data.Data` instance.

        Usage
        -----
        >>> data.pad()
        """
        if self.padded:
            return

        self.data, self.shape = self.pad_data(self.data, np.exp2(self.num_qubits))
        self.padded = True

    def to_quantumstate(self) -> None:
        """ Converts a `qickit.data.Data` instance to a quantum state.

        Usage
        -----
        >>> data.to_quantumstate()
        """
        if not self.normalized:
            self.normalize()

        if not self.padded:
            self.pad()

    def to_bra(
            self,
            data: NDArray[np.complex128]
        ) -> None:
        """ Convert the data to a bra vector.

        Parameters
        ----------
        `data` : NDArray[np.complex128]
            The data.

        Raises
        ------
        ValueError
            - If the data is a scalar or an operator.

        Usage
        -----
        >>> data = np.array([1, 2, 3, 4])
        >>> to_bra(data)
        """
        if data.ndim == 0:
            raise ValueError("Cannot convert a scalar to a bra.")
        elif data.ndim == 1:
            if data.shape[0] == 1:
                raise ValueError("Cannot convert a scalar to a bra.")
            else:
                self.data = data
                self.shape = self.data.shape
        elif data.ndim == 2:
            if data.shape[0] == 1:
                if data.shape[1] == 1:
                    raise ValueError("Cannot convert a scalar to a bra.")
                else:
                    self.data = data.reshape(1, -1)[0]
                    self.shape = self.data.shape
            else:
                raise ValueError("Cannot convert an operator to a bra.")
        else:
            raise ValueError("Cannot convert a N-dimensional array to a bra.")

        self.data = self.data.astype(np.complex128)

        # Normalize and pad the data to satisfy the quantum state requirements
        self.to_quantumstate()

    def to_ket(self) -> ket.Ket:
        """ Convert the bra vector to a ket vector.

        Returns
        -------
        qickit.primitives.Ket
            The ket vector.

        Usage
        -----
        >>> bra.to_ket()
        """
        return ket.Ket(self.data.conj().reshape(1, -1)[0])

    def compress(
            self,
            compression_percentage: float
        ) -> None:
        """ Compress a `qickit.data.Data` instance.

        Parameters
        ----------
        `compression_percentage` : float
            The percentage of compression.

        Usage
        -----
        >>> data.compress(50)
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
        """ Change the indexing of a `qickit.primitives.Bra` instance.

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
        >>> data.change_indexing("snake")
        """
        if index_type == "snake":
            if self.num_qubits >= 3:
                # Convert the bra vector to a matrix (image)
                self.data = self.data.reshape(2, -1)
                # Reverse the elements in odd rows
                self.data[1::2, :] = self.data[1::2, ::-1]

                self.data = self.data.flatten()
        elif index_type == "row":
            self.data = self.data
        else:
            raise ValueError("Index type not supported.")

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
            - If the the bra and operator are incompatible.
        NotImplementedError
            - If the `other` type is incompatible.
        """
        if isinstance(other, (SupportsFloat, complex)):
            return
        elif isinstance(other, ket.Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot contract two incompatible vectors.")
        elif isinstance(other, operator.Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible vectors.")
        else:
            raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __eq__(
            self,
            other: object
        ) -> bool:
        """ Check if two bra vectors are equal.

        Parameters
        ----------
        `other` : object
            The other bra vector.

        Returns
        -------
        bool
            Whether the two bra vectors are equal.

        Usage
        -----
        >>> bra1 = Bra(np.array([1+0j, 0+0j]))
        >>> bra2 = Bra(np.array([1+0j, 0+0j]))
        >>> bra1 == bra2
        """
        if isinstance(other, Bra):
            return bool(np.all(np.isclose(self.data, other.data, atol=1e-10, rtol=0)))

        raise NotImplementedError(f"Equality with {type(other)} is not supported.")

    def __len__(self) -> int:
        """ Return the length of the bra vector.

        Returns
        -------
        int
            The length of the bra vector.

        Usage
        -----
        >>> len(bra)
        """
        return len(self.data)

    def __add__(
            self,
            other: Bra
        ) -> Bra:
        """ Superpose two bra states together.

        Parameters
        ----------
        `other` : qickit.primitives.Bra
            The other bra state.

        Returns
        -------
        qickit.primitives.Bra
            The superposed bra state.

        Raises
        ------
        ValueError
            - If the two bra states are incompatible.

        Usage
        -----
        >>> bra1 = Bra(np.array([1+0j, 0+0j]))
        >>> bra2 = Bra(np.array([1+0j, 0+0j]))
        >>> bra1 + bra2
        """
        if isinstance(other, Bra):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot add two incompatible vectors.")
            return Bra((self.data + other.data).astype(np.complex128))

        raise NotImplementedError(f"Addition with {type(other)} is not supported.")

    @overload
    def __mul__(
            self,
            other: Scalar
        ) -> Bra:
        ...

    @overload
    def __mul__(
            self,
            other: ket.Ket
        ) -> Scalar:
        ...

    @overload
    def __mul__(
            self,
            other: operator.Operator
        ) -> Bra:
        ...

    def __mul__(
            self,
            other: Scalar | ket.Ket | operator.Operator
        ) -> Scalar | Bra:
        """ Multiply the bra by a scalar, a ket, or an operator.

        The multiplication of a bra with a ket is defined as:
        - ⟨ψ'|ψ⟩ = s, where s is a scalar

        The multiplication of a bra with an operator is defined as:
        - ⟨ψ|A = ⟨ψ'|

        Notes
        -----
        The multiplication of a bra with a scalar does not change the bra. This is because
        the norm of the bra is preserved, and the scalar is multiplied with each element of the
        bra. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : qickit.primitives.Scalar | qickit.primitives.Ket | qickit.primitives.Operator
            The other object to multiply the bra by.

        Returns
        -------
        qickit.primitives.Scalar | qickit.primitives.Bra
            The result of the multiplication.

        Raises
        ------
        ValueError
            - If the two vectors are incompatible.
            - If the operator dimensions are incompatible.
        NotImplementedError
            - If the `other` type is incompatible.

        Usage
        -----
        >>> scalar = 2
        >>> bra = Bra(np.array([1+0j, 0+0j]))
        >>> bra * scalar
        >>> bra = Bra(np.array([1+0j, 0+0j]))
        >>> ket = Ket(np.array([1+0j, 0+0j]))
        >>> bra * ket
        >>> bra = Bra(np.array([1+0j, 0+0j]))
        >>> operator = Operator([[1+0j, 0+0j],
        ...                      [0+0j, 1+0j]])
        >>> bra * operator
        """
        if isinstance(other, (SupportsFloat, complex)):
            return Bra((self.data * other).astype(np.complex128)) # type: ignore
        elif isinstance(other, ket.Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot contract two incompatible vectors.")
            return np.dot(self.data, other.data).flatten()[0]
        elif isinstance(other, operator.Operator):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot multiply two incompatible vectors.")
            return Bra(self.data @ other.data)
        else:
            raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __rmul__(
            self,
            other: Scalar
        ) -> Bra:
        """ Multiply the bra by a scalar.

        Notes
        -----
        The multiplication of a bra with a scalar does not change the bra. This is because
        the norm of the bra is preserved, and the scalar is multiplied with each element of the
        bra. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : qickit.primitives.Scalar
            The scalar to multiply the bra by.

        Returns
        -------
        qickit.primitives.Bra
            The bra multiplied by the scalar.

        Usage
        -----
        >>> scalar = 2
        >>> bra = Bra(np.array([1+0j, 0+0j]))
        >>> scalar * bra
        """
        if isinstance(other, (SupportsFloat, complex)):
            return Bra((self.data * other).astype(np.complex128)) # type: ignore

        raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __str__(self) -> str:
        """ Return the string representation of the bra vector.

        Returns
        -------
        str
            The string representation of the bra vector.

        Usage
        -----
        >>> str(bra)
        """
        return f"⟨{self.label}|"

    def __repr__(self) -> str:
        """ Return the string representation of the bra vector.

        Returns
        -------
        str
            The string representation of the bra vector.

        Usage
        -----
        >>> repr(bra)
        """
        return f"{self.__class__.__name__}(data={self.data}, label={self.label})"