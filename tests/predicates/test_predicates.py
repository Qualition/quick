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

__all__ = [
    "test_is_square_matrix",
    "test_is_diagonal_matrix",
    "test_is_symmetric_matrix",
    "test_is_identity_matrix",
    "test_is_unitary_matrix",
    "test_is_hermitian_matrix",
    "test_is_positive_semidefinite_matrix",
    "test_is_isometry"
]

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.stats import unitary_group

from qickit.predicates import (
    is_square_matrix,
    is_diagonal_matrix,
    is_symmetric_matrix,
    is_identity_matrix,
    is_unitary_matrix,
    is_hermitian_matrix,
    is_positive_semidefinite_matrix,
    is_isometry
)

@pytest.mark.parametrize("array, expected", [
    (np.random.rand(2, 2), True),
    (np.random.rand(3, 3), True),
    (np.random.rand(4, 4), True),
    (np.random.rand(5, 5), True),
    (np.random.rand(2, 1), False),
    (np.random.rand(1, 3), False),
    (np.random.rand(4, 12), False)
])
def test_is_square_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_square_matrix()` method.
    """
    assert is_square_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (np.diag([1, 2, 3]), True),
    (np.diag([4, 5, 6, 7]), True),
    (np.diag([8, 9]), True),
    (np.array([
        [1, 2, 0],
        [0, 3, 4],
        [0, 0, 5]
    ]), False),
    (np.array([
        [1, 0, 0],
        [2, 3, 0],
        [0, 0, 4]
    ]), False),
    (np.array([
        [1, 0],
        [1, 1]
    ]), False)
])
def test_is_diagonal_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_diagonal_matrix()` method with diagonal matrices.
    """
    assert is_diagonal_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ]), True),
    (np.array([
        [1, 2, 3, 4],
        [2, 5, 6, 7],
        [3, 6, 8, 9],
        [4, 7, 9, 10]
    ]), True),
    (np.array([
        [1, 2],
        [2, 3]
    ]), True),
    (np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]), False),
    (np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]), False),
    (np.array([
        [1, 2],
        [3, 4]
    ]), False)
])
def test_is_symmetric_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_symmetric_matrix()` method with symmetric matrices.
    """
    assert is_symmetric_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (np.eye(2), True),
    (np.eye(3), True),
    (np.eye(4), True),
    (np.eye(5), True),
    (np.random.rand(2, 2), False),
    (np.random.rand(3, 3), False),
    (np.random.rand(4, 4), False),
    (np.random.rand(3, 4), False)
])
def test_is_identity_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_identity_matrix()` method with identity matrices.
    """
    assert is_identity_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (unitary_group.rvs(2), True),
    (unitary_group.rvs(3), True),
    (unitary_group.rvs(4), True),
    (unitary_group.rvs(5), True),
    (np.random.rand(2, 2), False),
    (np.random.rand(3, 3), False),
    (np.random.rand(3, 4), False),
    (np.random.rand(5, 2), False)
])
def test_is_unitary_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_unitary_matrix()` method.
    """
    assert is_unitary_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (np.array([
        [1, 2 + 1j, 3],
        [2 - 1j, 4, 5 + 2j],
        [3, 5 - 2j, 6]
    ]), True),
    (np.array([
        [1, 2 + 1j],
        [2 - 1j, 3]
    ]), True),
    (np.array([
        [1, 0],
        [0, 1]
    ]), True),
    (np.array([
        [1, 2 + 1j, 3],
        [2 + 1j, 4, 5 + 2j],
        [3, 5 - 2j, 6]
    ]), False),
    (np.array([
        [1, 2 + 1j],
        [2 + 1j, 3]
    ]), False),
    (np.array([
        [1, 2],
        [3, 4]
    ]), False)
])
def test_is_hermitian_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_hermitian_matrix()` method with Hermitian matrices.
    """
    assert is_hermitian_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (np.array([
        [1, 0],
        [0, 1]
    ]), True),
    (np.array([
        [1, 0],
        [0, 0]
    ]), True),
    (np.array([
        [1, 2],
        [2, 3]
    ]), False),
    (np.array([
        [1, 2],
        [3, 4]
    ]), False),
    (np.array([
        [1, 0],
        [0, -1]
    ]), False),
    (np.array([
        [1, 2],
        [2, 1]
    ]), False)
])
def test_is_positive_semidefinite_matrix(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `.is_positive_semidefinite_matrix()` method with positive semidefinite matrices.
    """
    assert is_positive_semidefinite_matrix(array) == expected

@pytest.mark.parametrize("array, expected", [
    (np.array([
        [0.56078693+0.13052803j, -0.31583062-0.08879493j],
        [0.7123732 +0.2419316j, 0.39227097-0.22401521j],
        [0.30203025+0.10607406j, -0.38000351+0.7374988j]
    ], dtype=np.complex128), True),
    (np.array([
        [0.08849653+0.24435482j],
        [-0.72166734+0.64160373j]
    ], dtype=np.complex128), True),
    (np.array([
        [0.02572621+0.08711405j, -0.84637795+0.52477964j],
        [0.86175428-0.4991281j, -0.06470557-0.06374861j]
    ]), True),
    (np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.complex128), False),
    (np.random.rand(3, 3), False),
    (np.random.rand(1, 2), False)
])
def test_is_isometry(
        array: NDArray[np.complex128],
        expected: bool
    ) -> None:
    """ Test the `is_isometry` function with various matrices.
    """
    assert is_isometry(array) == expected