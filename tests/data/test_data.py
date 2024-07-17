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

__all__ = ["TestData"]

import copy
from collections.abc import MutableSequence
import numpy as np
import pytest

# QICKIT imports
from qickit.data import Data


class TestData:
    """ Test the `qickit.data.Data` class.
    """
    def test_init(self) -> None:
        """ Test the initialization of the `qickit.data.Data` class.
        """
        # Test the initialization of a vector
        data = Data([1, 2, 3])
        np_data = Data(np.array([1, 2, 3]))
        assert np_data == data

        # Test the initialization of a matrix
        data = Data([[1, 2, 3], [4, 5, 6]])
        np_data = Data(np.array([[1, 2, 3], [4, 5, 6]]))
        assert np_data == data

        with pytest.raises(TypeError):
            Data("not a list") # type: ignore

        with pytest.raises(TypeError):
            Data(1) # type: ignore

    @staticmethod
    def is_normalized(state: MutableSequence[MutableSequence[float]] | MutableSequence[float]) -> bool:
        """ Test if a state vector is normalized.

        Parameters
        -----------
        state (MutableSequence[MutableSequence[float]] | MutableSequence[float]):
            A list representing a state vector.

        Returns
        --------
        (bool): True if the state vector is normalized. False otherwise.
        """
        # Calculate the norm squared of the state vector
        norm_squared = np.linalg.norm(state) ** 2

        # Set the tolerance
        epsilon = 1e-6

        # Assert the Born rule
        if np.abs(norm_squared - 1) < epsilon:
            return True
        else:
            return False

    def test_normalize(self) -> None:
        """ Test the `.normalize()` method.
        """
        data = Data([[1, 2, 3], [4, 5, 6]])
        assert not data.normalized

        data.normalize()
        assert data.normalized
        assert self.is_normalized(data.data)

        # Call the `.normalize()` method again to cover the case when the
        # data is already normalized (idempotent property)
        new_data = copy.deepcopy(data)
        new_data.normalize()
        assert new_data == data

    def test_denormalize(self) -> None:
        """ Test the `.denormalize()` method.
        """
        data = Data([[1, 2, 3], [4, 5, 6]])
        data.normalize()
        assert data.normalized

        data.denormalize()
        assert not data.normalized
        assert data == [[1, 2, 3], [4, 5, 6]]

        # Call the `.denormalize()` method again to cover the case when the
        # data is already denormalized (idempotent property)
        new_data = copy.deepcopy(data)
        new_data.denormalize()
        assert new_data == data

    def test_pad_vector(self) -> None:
        """ Test the `.pad()` method for a vector.
        """
        vector = Data([1, 2, 3])
        assert not vector.padded

        vector.pad()
        assert vector.padded
        assert vector.shape == (4,)

        # Call the `.pad()` method again to cover the case when the
        # data is already padded (idempotent property)
        new_vector = copy.deepcopy(vector)
        new_vector.pad()
        assert new_vector == vector

    def test_pad_matrix(self) -> None:
        """ Test the `.pad()` method for a matrix.
        """
        matrix = Data([[1, 2, 3], [4, 5, 6]])
        assert not matrix.padded

        matrix.pad()
        assert matrix.padded
        assert matrix.shape == (2, 4)

        # Call the `.pad()` method again to cover the case when the
        # data is already padded
        new_matrix = copy.deepcopy(matrix)
        new_matrix.pad()
        assert new_matrix == matrix

    def test_check_padding_vector(self) -> None:
        """ Test the `.check_padding()` method for a vector that is padded.
        """
        vector = np.array([1, 2, 3, 4])
        assert Data.check_padding(vector)

    def test_check_padding_vector_fail(self) -> None:
        """ Test the `.check_padding()` method for a vector that is not padded.
        """
        vector = np.array([1, 2, 3])
        assert not Data.check_padding(vector)

    def test_check_padding_matrix(self) -> None:
        """ Test the `.check_padding()` method for a matrix that is padded.
        """
        matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert Data.check_padding(matrix)

    def test_check_padding_matrix_fail(self) -> None:
        """ Test the `.check_padding()` method for a matrix that is not padded.
        """
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        assert not Data.check_padding(matrix)

    def test_check_padding_scalar(self) -> None:
        """ Test the `.check_padding()` method for a scalar.
        """
        scalar = np.array(1)
        with pytest.raises(ValueError):
            Data.check_padding(scalar)

    def test_compress(self) -> None:
        """ Test the `.compress()` method.
        """
        pass

    def test_to_quantumstate(self) -> None:
        """ Test the `.to_quantumstate()` method.
        """
        pass

    def test_iscloseto_identical(self) -> None:
        """ Test the `Data.iscloseto()` method when the data instances are identical.
        """
        data1 = Data([1, 2, 3])
        data2 = Data([1, 2, 3])
        assert Data.iscloseto(data1, data2)

    def test_iscloseto_different(self) -> None:
        """ Test the `Data.iscloseto()` method when the data instances are different.
        """
        data1 = Data([1, 2, 3])
        data2 = Data([4, 5, 6])
        assert not Data.iscloseto(data1, data2)

    def test_iscloseto_approximate(self) -> None:
        """ Test the `Data.iscloseto()` method when the data instances are approximately equal.
        """
        data1 = Data([1, 2, 3])
        data2 = Data([1, 2, 3.0000001])
        assert not Data.iscloseto(data1, data2)
        assert Data.iscloseto(data1, data2, 1e-7)

    def test_iscloseto_different_types(self) -> None:
        """ Test the `Data.iscloseto()` method when the data instances are of different container types.
        """
        data1 = [1, 2, 3]
        data2 = Data([1, 2, 3])
        assert Data.iscloseto(data1, data2)

    def test_iscloseto_wrong_type(self) -> None:
        """ Test the `Data.iscloseto()` method when the data instances are of the wrong type.
        """
        data1 = "not a data instance"
        data2 = Data([1, 2, 3])
        with pytest.raises(TypeError):
            Data.iscloseto(data1, data2)

    def test_eq_identical(self) -> None:
        """ Test the `__eq__()` method when the data instances are identical.
        """
        data1 = Data([1, 2, 3])
        data2 = Data([1, 2, 3])
        assert data1 == data2

    def test_eq_different(self) -> None:
        """ Test the `__eq__()` method when the data instances are different.
        """
        data1 = Data([1, 2, 3])
        data2 = Data([4, 5, 6])
        assert data1 != data2

    def test_eq_approximate(self) -> None:
        """ Test the `__eq__()` method when the data instances are approximately equal.
        """
        data1 = Data([1, 2, 3])
        data2 = Data([1, 2, 3.00001])
        assert data1 != data2

    def test_eq_different_types(self) -> None:
        """ Test the `__eq__()` method when the data instances are of different container types.
        """
        data1 = [1, 2, 3]
        data2 = Data([1, 2, 3])
        data1 == data2 # type: ignore

    def test_eq_wrong_type(self) -> None:
        """ Test the `__eq__()` method when the data instances are of the wrong type.
        """
        data1 = "not a data instance"
        data2 = Data([1, 2, 3])
        with pytest.raises(TypeError):
            data1 == data2 # type: ignore

    def test_len_vector(self) -> None:
        """ Test the `__len__()` method for a vector.
        """
        data = Data([1, 2, 3])
        assert len(data) == 3

    def test_len_matrix(self) -> None:
        """ Test the `__len__()` method for a matrix.
        """
        data = Data([[1, 2, 3], [4, 5, 6]])
        assert len(data) == 6

    def test_mul(self) -> None:
        """ Test the `__mul__()` method.
        """
        data = Data([1, 2, 3])
        assert data * 2 == Data([2, 4, 6])
        assert data * 2.5 == Data([2.5, 5, 7.5])

    def test_rmul(self) -> None:
        """ Test the `__rmul__()` method.
        """
        data = Data([1, 2, 3])
        assert 2 * data == Data([2, 4, 6])
        assert 2.5 * data == Data([2.5, 5, 7.5])

    def test_repr(self) -> None:
        """ Test the `__repr__()` method.
        """
        data = Data([1, 2, 3])
        assert repr(data) == "Data(data=[1 2 3])"

        data = Data([[1, 2, 3], [4, 5, 6]])
        assert repr(data) == "Data(data=[[1 2 3]\n [4 5 6]])"