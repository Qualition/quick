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

""" Ket vector class for representing ket states.
"""

from __future__ import annotations

__all__ = ["Ket"]

import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal, overload, SupportsFloat, TypeAlias

import qickit.primitives.operator as operator
import qickit.primitives.bra as bra

# `Scalar` is a type alias that represents a scalar value that can be either
# a real number or a complex number.
Scalar: TypeAlias = SupportsFloat | complex


class Ket:
    """ `qickit.primitives.Ket` is a class that represents a quantum ket vector. Ket vectors are
    complex, column vectors with a magnitude of 1 which represent quantum states. The ket vectors are
    the complex conjugates of the bra vectors.

    Parameters
    ----------
    `data` : NDArray[np.complex128]
        The ket vector data. The data will be normalized to 2-norm and padded if necessary.
    `label` : str, optional
        The label of the ket vector.

    Attributes
    ----------
    `label` : str, optional, default="Ψ"
        The label of the ket vector.
    `data` : NDArray[np.complex128]
        The ket vector data.
    `norm_scale` : np.float64
        The normalization scale.
    `normalized` : bool
        Whether the ket vector is normalized to 2-norm or not.
    `shape` : Tuple[int, int]
        The shape of the ket vector.
    `num_qubits` : int
        The number of qubits represented by the ket vector.

    Raises
    ------
    ValueError
        - If the data is a scalar or an operator.

    Usage
    -----
    >>> data = np.array([1, 2, 3, 4])
    >>> ket = Ket(data)
    """
    def __init__(
            self,
            data: NDArray[np.complex128],
            label: str | None = None
        ) -> None:
        """ Initialize a `qickit.primitives.Ket` instance.
        """
        if label is None:
            self.label = "\N{GREEK CAPITAL LETTER PSI}"
        else:
            self.label = label

        self.norm_scale = np.linalg.norm(data.flatten())
        self.data = data
        self.shape = data.shape
        self.num_qubits = int(np.ceil(np.log2(self.shape[0])))
        self.is_normalized()
        self.is_padded()
        self.to_ket(data)

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
        """ Normalize a `qickit.primitives.Ket` instance to 2-norm.
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
        >>> data = np.array([[1, 2], [3, 4]])
        >>> pad_data(data)
        """
        flattened_data = data.flatten()

        padded_data = np.pad(
            flattened_data, (0, int(target_size - len(flattened_data))),
            mode="constant"
        ).reshape(-1, 1)

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

    def to_ket(
            self,
            data: NDArray[np.complex128]
        ) -> None:
        """ Convert the data to a ket vector.

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
        >>> ket.to_ket(data)
        """
        if data.ndim == 0:
            raise ValueError("Cannot convert a scalar to a ket.")
        elif data.ndim == 1:
            if data.shape[0] == 1:
                raise ValueError("Cannot convert a scalar to a ket.")
            else:
                self.data = data.reshape(-1, 1)
        elif data.ndim == 2:
            if data.shape[1] == 1:
                if data.shape[0] == 1:
                    raise ValueError("Cannot convert a scalar to a ket.")
                else:
                    self.data = data
            else:
                raise ValueError("Cannot convert an operator to a ket.")
        else:
            raise ValueError("Cannot convert a N-dimensional array to a ket.")

        self.data = self.data.astype(np.complex128)

        # Normalize and pad the data to satisfy the quantum state requirements
        self.to_quantumstate()

    def to_bra(self) -> bra.Bra:
        """ Convert the ket to a bra.

        Returns
        -------
        qickit.primitives.Bra
            The bra vector.

        Usage
        -----
        >>> ket.to_bra()
        """
        return bra.Bra(self.data.conj().reshape(1, -1)) # type: ignore

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
        flattened_data = self.data.flatten()
        data_sort_ind = np.argsort(np.abs(flattened_data))

        # Set the smallest absolute values of data to zero according to compression parameter
        cutoff = int((compression_percentage / 100.0) * len(flattened_data))
        for i in data_sort_ind[:cutoff]:
            flattened_data[i] = 0

        self.data = flattened_data.reshape(-1, 1)

    def change_indexing(
            self,
            index_type: Literal["row", "snake"]
        ) -> None:
        """ Change the indexing of a `qickit.primitives.Ket` instance.

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

                self.data = self.data.flatten().reshape(-1, 1)
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
        NotImplementedError
            - If the `other` type is incompatible.
        """
        if isinstance(other, (SupportsFloat, complex)):
            return
        elif isinstance(other, bra.Bra):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot contract two incompatible vectors.")
        elif isinstance(other, Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot contract two incompatible vectors.")
        else:
            raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __eq__(
            self,
            other: object
        ) -> bool:
        """ Check if two ket vectors are equal.

        Parameters
        ----------
        `other` : object
            The other ket vector.

        Returns
        -------
        bool
            Whether the two ket vectors are equal.

        Usage
        -----
        >>> ket1 = Ket(np.array([1+0j, 0+0j]))
        >>> ket2 = Ket(np.array([1+0j, 0+0j]))
        >>> ket1 == ket2
        """
        if isinstance(other, Ket):
            return bool(np.all(np.isclose(self.data.flatten(), other.data.flatten(), atol=1e-10, rtol=0)))

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
        return len(self.data.flatten())

    def __add__(
            self,
            other: Ket
        ) -> Ket:
        """ Superpose two ket states together.

        Parameters
        ----------
        `other` : qickit.primitives.Ket
            The other ket state.

        Returns
        -------
        qickit.primitives.Ket
            The superposed ket state.

        Raises
        ------
        NotImplementedError
            - If the two vectors are incompatible.
        ValueError
            - If the two ket states are incompatible.

        Usage
        -----
        >>> ket1 = Ket(np.array([1+0j, 0+0j]))
        >>> ket2 = Ket(np.array([1+0j, 0+0j]))
        >>> ket3 = ket1 + ket2
        """
        if isinstance(other, Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot add two incompatible vectors.")
            return Ket((self.data.flatten() + other.data.flatten()).astype(np.complex128))

        raise NotImplementedError(f"Addition with {type(other)} is not supported.")

    @overload
    def __mul__(
            self,
            other: Scalar
        ) -> Ket:
        ...

    @overload
    def __mul__(
            self,
            other: bra.Bra
        ) -> operator.Operator:
        ...

    @overload
    def __mul__(
            self,
            other: Ket
        ) -> Ket:
        ...

    def __mul__(
            self,
            other: Scalar | bra.Bra | Ket
        ) -> Ket | operator.Operator:
        """ Multiply the ket by a scalar, bra, or ket.

        The multiplication of a ket with a bra is defined as:
        |ψ⟩⟨ψ|, which is called the projection operator and is implemented using the measurement
        operator.

        The multiplication of a ket with a ket is defined as:
        |ψ⟩⊗|ψ'⟩, which is called the tensor product of two quantum states.

        Notes
        -----
        The multiplication of a ket with a scalar does not change the ket. This is because
        the norm of the ket is preserved, and the scalar is multiplied with each element of the
        ket. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : qickit.primitives.Scalar | qickit.primitives.Bra | qickit.primitives.Ket
            The object to multiply the ket by.

        Returns
        -------
        qickit.primitives.Ket | qickit.primitives.Operator
            The ket or operator resulting from the multiplication.

        Raises
        ------
        ValueError
            - If the two vectors are incompatible.
        NotImplementedError
            - If the `other` type is incompatible.

        Usage
        -----
        >>> scalar = 2
        >>> ket = Ket(np.array([1+0j, 0+0j]))
        >>> ket = ket * scalar
        >>> bra = Bra(np.array([1+0j, 0+0j]))
        >>> ket = Ket(np.array([1+0j, 0+0j]))
        >>> operator = ket * bra
        >>> ket1 = Ket(np.array([1+0j, 0+0j]))
        >>> ket2 = Ket(np.array([1+0j, 0+0j]))
        >>> ket3 = ket1 * ket2
        """
        if isinstance(other, (SupportsFloat, complex)):
            return Ket((self.data * other).astype(np.complex128)) # type: ignore
        elif isinstance(other, bra.Bra):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot contract two incompatible vectors.")
            return operator.Operator(np.outer(self.data, other.data.conj()))
        elif isinstance(other, Ket):
            if self.num_qubits != other.num_qubits:
                raise ValueError("Cannot contract two incompatible vectors.")
            return Ket(np.kron(self.data, other.data))
        else:
            raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __rmul__(
            self,
            other: Scalar
        ) -> Ket:
        """ Multiply the ket by a scalar.

        Notes
        -----
        The multiplication of a ket with a scalar does not change the ket. This is because
        the norm of the ket is preserved, and the scalar is multiplied with each element of the
        ket. We provide the scalar multiplication for completeness.

        Parameters
        ----------
        `other` : qickit.primitives.Scalar
            The scalar to multiply the ket by.

        Returns
        -------
        qickit.primitives.Ket
            The ket multiplied by the scalar.

        Usage
        -----
        >>> scalar = 2
        >>> ket = Ket(np.array([1+0j, 0+0j]))
        >>> ket = scalar * ket
        """
        if isinstance(other, (SupportsFloat, complex)):
            return Ket((self.data * other).astype(np.complex128)) # type: ignore

        raise NotImplementedError(f"Multiplication with {type(other)} is not supported.")

    def __str__(self) -> str:
        """ Return the string representation of the ket.

        Returns
        -------
        str
            The string representation of the ket.

        Usage
        -----
        >>> ket = Ket(np.array([1+0j, 0+0j]))
        >>> str(ket)
        """
        return f"|{self.label}⟩"

    def __repr__(self) -> str:
        """ Return the string representation of the ket.

        Returns
        -------
        str
            The string representation of the ket.

        Usage
        -----
        >>> ket = Ket(np.array([1+0j, 0+0j]))
        >>> repr(ket)
        """
        return f"{self.__class__.__name__}(data={self.data}, label={self.label})"