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

__all__ = ['Data']

import numpy as np
from numpy.typing import NDArray
from PIL import Image as Img # type: ignore
import matplotlib.pyplot as plt

# Import `qickit.types.collection.Collection`
from qickit.types import Collection, NestedCollection

# Define the type alias for numbers
NumberType = int | float | complex


class Data:
    """ `qickit.data.Data` is the class used in Qickit's framework for accessing
    and manipulating data with built-in features such as normalization, and padding.

    Parameters
    ----------
    `data` : NestedCollection[NumberType]
        The datapoint values.
    `norm_scale` : float
        The normalization scale.
    `normalized` : bool
        Whether the Data is normalized to 2 norm.
    `padded` : bool
        Whether the Data is padded to a power of 2.
    `shape` : tuple[int, ...]
        The shape of the datapoint.

    Usage
    -----
    >>> data = Data([[1, 2], [3, 4]])
    """
    def __init__(self,
                 data: NestedCollection[NumberType]) -> None:
        """ Initialize a `qickit.data.Data` instance.
        """
        # Convert the data to `np.ndarray`
        if not isinstance(data, np.ndarray):
            self.data: NDArray = np.array(data)
        else:
            self.data = data

        # Save the data shape
        self.shape = self.data.shape

        # Set the norm scale (for normalization and denormalization)
        self.norm_scale = np.linalg.norm(self.data.flatten())

        # Set the normalized status
        self.is_normalized()

        # Set the padded status
        self.is_padded()

    @staticmethod
    def check_normalization(data: NDArray[np.number]) -> bool:
        """ Check if a data is normalized to 2-norm.

        Parameters
        ----------
        `data` : NDArray[np.number]
            The data.

        Returns
        -------
        bool
            Whether the vector is normalized to 2-norm or not.

        Usage
        -----
        >>> data = np.array([[1, 2], [3, 4]])
        >>> check_normalization(data)
        """
        # Flatten the data in case it is a 2d array
        data_vector = data.flatten()

        # Check whether the data is normalized to 2-norm
        sum_check = np.sum(data_vector**2)

        # Check if the sum of squared of the data elements is equal to
        # 1 with 1e-8 tolerance
        return bool(np.isclose(sum_check, 1.0, atol=1e-08))

    def is_normalized(self) -> None:
        """ Check if a `qickit.data.Data` instance is normalized to 2-norm.

        Usage
        -----
        >>> data.is_normalized()
        """
        # If data is normalized, set `normalized` to True, Otherwise,
        # set `normalized` False
        self.normalized = self.check_normalization(self.data)

    @staticmethod
    def normalize_data(data: NDArray[np.number],
                       norm_scale: np.float64) -> NDArray[np.number]:
        """ Normalize the data to 2-norm, and return the
        normalized data.

        Parameters
        ----------
        `data` : NDArray[np.number]
            The data.
        `norm_scale` : np.float64
            The normalization scale.

        Returns
        -------
        NDArray[np.number]
            The 2-norm normalized data.

        Usage
        -----
        >>> data = np.array([[1, 2], [3, 4]])
        >>> norm_scale = np.linalg.norm(data.flatten())
        >>> data_shape = data.shape
        >>> normalize_data(data, norm_scale, data_shape)
        """
        # Normalize the vector to 2-norm
        normalized_vector = np.multiply(data, 1/norm_scale)

        return normalized_vector

    def normalize(self) -> None:
        """ Normalize a `qickit.data.Data` instance to 2-norm.
        """
        # If the data is already normalized
        if self.normalized:
            return

        # If the data is not normalized, normalize the data
        self.data = self.normalize_data(self.data, self.norm_scale)

        # Set normalized to True
        self.normalized = True

    @staticmethod
    def denormalize_data(data: NDArray[np.number],
                         norm_scale: np.float64) -> NDArray[np.number]:
        """ Denormalize the data from 2-norm, and return the
        denormalized data.

        Parameters
        ----------
        `data` : NDArray[np.number]
            The 2-norm normalized datapoint.
        `norm_scale` : np.float64
            The normalization scale.

        Returns
        -------
        NDArray[np.number]
            The denormalized data.
        """
        # Denormalize the vector by applying the inverse of the
        # normalization factor
        denormalized_vector = np.multiply(data, norm_scale)

        return denormalized_vector

    def denormalize(self) -> None:
        """ Denormalize a `qickit.data.Data` instance from 2-norm.
        """
        # If the data is already denormalized, or simply not normalized
        if self.normalized is False:
            return

        # If the data is normalized, denormalize the data
        self.data = self.denormalize_data(self.data, self.norm_scale)

        # Set normalized to False
        self.normalized = False

    @staticmethod
    def check_padding(data: NDArray[np.number]) -> bool:
        """ Check if a data is normalized to 2-norm.

        Parameters
        ----------
        `data` : NDArray[np.number]
            The data.

        Returns
        -------
        bool
            Whether the vector is normalized to 2-norm or not.
        """
        # If the data is a vector
        if data.ndim == 1:
            # Check if the length of the vector is a power of 2
            return bool(len(data) == np.exp2(np.ceil(np.log2(len(data)))))

        # If the data is a matrix
        elif data.ndim == 2:
            # Define the rows and columns size
            rows, cols = data.shape
            # Calculate the target size
            target_size = np.exp2(np.ceil(np.log2(rows * cols)))
            # Check if the number of matrix elements is a power of 2,
            # set `padded` to True
            return bool(rows == np.ceil(np.divide(target_size, cols)))

        else:
            raise ValueError("Data must be a vector or a matrix.")

    def is_padded(self) -> None:
        """ Check if a `qickit.data.Data` instance is padded to a power of 2.
        """
        self.padded = self.check_padding(self.data)

    @staticmethod
    def pad_data(data: NDArray[np.number]) -> tuple[NDArray[np.number],
                                                    tuple[int, ...]]:
        """ Pad data with zeros up to the nearest power of 2, and return
        the padded data.

        Parameters
        ----------
        `data` : NDArray[np.number]
            The data to be padded.

        Returns
        -------
        `padded_data` : NDArray[np.number]
            The padded data.
        `data_shape` : (tuple[int, ...])
            The updated shape.
        """
        # If the data is a vector
        if data.ndim == 1:
            # Calculate the target size
            target_size = np.exp2(np.ceil(np.log2(len(data))))

            # Pad the vector with 0s
            padded_data = np.pad(data, (0, target_size - len(data)), mode='constant')

            # Update data shape
            updated_shape = padded_data.shape

            return padded_data, updated_shape

        # If the data is a matrix
        elif data.ndim == 2:
            # Define the rows and columns size
            rows, cols = data.shape

            # Calculate the target size
            target_size = np.exp2(np.ceil(np.log2(rows * cols)))

            # Calculate the number of rows and columns needed for the
            # padded matrix
            target_rows = np.divide(target_size, cols)
            target_cols = np.divide(target_size, rows)

            # Ensure the target rows and columns are integers (i.e., we
            # can't have 1.5 rows)
            if target_rows != int(target_rows) or target_cols != int(target_cols):
                # If the data is wider than tall
                if cols > rows:
                    # Calculate the number of columns needed for the
                    # padded matrix (prioritize wider images)
                    cols = int(np.divide(target_size, rows))

                else:
                    # Calculate the number of rows needed for the
                    # padded matrix (prioritize taller images)
                    rows = int(np.divide(target_size, cols))

                # If we cannot pad, we stretch the whole matrix to the
                # nearest power of 2
                resized_data = np.asarray(Img.fromarray(data).resize((cols, rows), 0))

                return resized_data, resized_data.shape

            # Ensure the target rows and columns are int type
            target_rows = int(target_rows)
            target_cols = int(target_cols)

            # Pad the matrix with 0s
            if np.multiply(target_rows, cols) == target_size:
                padded_data = np.zeros((target_rows, cols), dtype=data.dtype)
            else:
                padded_data = np.zeros((rows, target_cols), dtype=data.dtype)

            # Copy the original matrix into the top-left corner of the
            # padded matrix
            padded_data[:rows, :cols] = data

            # Update data shape
            updated_shape = padded_data.shape

            return padded_data, updated_shape

        else:
            raise ValueError("Data must be a vector or a matrix.")

    def pad(self) -> None:
        """ Pad a `qickit.data.Data` instance.
        """
        # Check if the data is already padded or not
        if self.padded:
            return

        # Pad the data
        self.data, self.shape = self.pad_data(self.data)

        # Set padded to True
        self.padded = True

    def to_quantumstate(self) -> None:
        """ Converts a `qickit.data.Data` instance to a quantum state.
        """
        if self.normalized is False:
            # Normalize the data
            self.normalize()

        if self.padded is False:
            # Pad the data
            self.pad()

    def compress(self,
                 compression_percentage: float) -> None:
        """ Compress a `qickit.data.Data` instance.

        Parameters
        ----------
        `compression_percentage` : float
            The percentage of compression.
        """
        # Flatten the data in case it is not 1-dimensional
        data = self.data.flatten()

        # Sort the data
        data_sort_ind = np.argsort(np.abs(data))

        # Set the smallest absolute values of data to zero according to compression parameter
        cutoff = int((compression_percentage / 100.0) * len(data))
        for i in data_sort_ind[:cutoff]:
            data[i] = 0

        # Reshape the data
        self.data = data.reshape(self.shape)

    def draw(self) -> None:
        """ Draw the `qickit.data.Data` instance.
        """
        plt.imshow(self.data)
        plt.show()

    @staticmethod
    def iscloseto(first_data: object,
                  second_data: object,
                  tolerance: float = 1e-8) -> bool:
        """ Check if two `qickit.data.Data` instances are close to each other.

        Parameters
        ----------
        `first_data` : object
            The first data to compare with.
        `second_data` : object
            The second data to compare with.
        `tolerance` : float
            The tolerance for the comparison.

        Returns
        -------
        bool
            Whether the two `qickit.Data` instances are close to each other or not.
        """
        # Define a `qickit.Data` instance
        if not isinstance(first_data, Data):
            if not isinstance(first_data, Collection):
                raise TypeError("Data must be a `qickit.Data` instance or a Collection.")
            else:
                first_data = Data(first_data)

        # Define a `qickit.Data` instance
        if not isinstance(second_data, Data):
            if not isinstance(second_data, Collection):
                raise TypeError("Data must be a `qickit.Data` instance or a Collection.")
            else:
                second_data = Data(second_data)

        return bool(np.allclose(first_data.data, second_data.data, tolerance))

    def change_indexing(self,
                        index_type: str) -> None:
        """ Change the indexing of a `qickit.data.Data` instance.

        Parameters
        ----------
        `index_type` : str
            The new indexing type.
        """
        if index_type == 'snake':
            # Ensure the array has two dimensions
            if len(self.shape) != 2:
                raise ValueError("Data array must be two-dimensional.")

            # Reverse the elements in odd rows
            self.data[1::2, :] = self.data[1::2, ::-1]

        elif index_type == 'row':
            self.data = self.data

        else:
            raise ValueError("Index type not supported.")

    def __repr__(self) -> str:
        """ Return a string representation of the `qickit.data.Data` instance.

        Returns
        -------
        str
            The string representation of the `qickit.data.Data` instance.
        """
        return f'Data(data={self.data})'

    def __eq__(self,
               other_data: object) -> bool:
        """ Check if two `qickit.data.Data` instances are equal.

        Parameters
        ----------
        `other_data` : qickit.data.Data
            The other data to compare with.

        Returns
        -------
        bool
            Whether the two `qickit.Data` instances are equal or not.
        """
        if not isinstance(other_data, Data):
            if not isinstance(other_data, Collection):
                raise TypeError("Data must be a `qickit.Data` instance or a Collection.")
            else:
                other_data = Data(other_data)

        return bool(np.all(self.data == other_data.data))

    def __len__(self) -> int:
        """ Return the length of the `qickit.data.Data` instance.

        Returns
        -------
        int
            The length of the `qickit.data.Data` instance.
        """
        return len(self.data.flatten())

    def __mul__(self,
                multiplier: float) -> None:
        """ Multiply a `qickit.data.Data` instance by a scalar.

        Parameters
        ----------
        `multiplier` : float
            The scalar to multiply by.
        """
        # Multiply the data by the multiplier
        self.data = np.multiply(self.data, multiplier)