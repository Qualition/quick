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

""" Predicates module.
"""

from __future__ import annotations

__all__ = [
    "is_square_matrix",
    "is_diagonal_matrix",
    "is_symmetric_matrix",
    "is_identity_matrix",
    "is_unitary_matrix",
    "is_hermitian_matrix",
    "is_positive_semidefinite_matrix",
    "is_isometry"
]

import numpy as np
from numpy.typing import NDArray

ATOL_DEFAULT = 1e-8
RTOL_DEFAULT = 1e-5


def is_square_matrix(matrix: NDArray[np.complex128]) -> bool:
    """ Test if an array is a square matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.

    Returns
    -------
    bool
        True if the matrix is square, False otherwise.

    Usage
    -----
    >>> is_square_matrix(np.eye(2))
    """
    if matrix.ndim != 2:
        return False
    shape = matrix.shape
    return shape[0] == shape[1]

def is_diagonal_matrix(
        matrix: NDArray[np.complex128],
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a diagonal matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is a diagonal matrix, False otherwise.

    Usage
    -----
    >>> is_diagonal_matrix(np.eye(2))
    """
    if not is_square_matrix(matrix):
        return False

    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT

    if matrix.ndim != 2:
        return False
    return np.allclose(matrix, np.diag(np.diagonal(matrix)), rtol=rtol, atol=atol)

def is_symmetric_matrix(
        matrix: NDArray[np.complex128],
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a symmetric matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is symmetric, False otherwise.

    Usage
    -----
    >>> is_symmetric_matrix(np.eye(2))
    """
    if not is_square_matrix(matrix):
        return False

    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT

    if matrix.ndim != 2:
        return False
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

def is_identity_matrix(
        matrix: NDArray[np.complex128],
        ignore_phase: bool=False,
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is an identity matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `ignore_phase` : bool, optional, default=False
        If True, ignore the phase of the matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is an identity matrix, False otherwise.

    Usage
    -----
    >>> is_identity_matrix(np.eye(2))
    """
    if not is_square_matrix(matrix):
        return False

    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    if matrix.ndim != 2:
        return False
    if ignore_phase:
        # If the matrix is equal to an identity up to a phase, we can
        # remove the phase by multiplying each entry by the complex
        # conjugate of the phase of the [0, 0] entry.
        theta = np.angle(matrix[0, 0])
        matrix = np.exp(-1j * theta) * matrix

    # Check if square identity
    identity = np.eye(len(matrix))
    return np.allclose(matrix, identity, rtol=rtol, atol=atol)

def is_unitary_matrix(
        matrix: NDArray[np.complex128],
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a unitary matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is a unitary matrix, False otherwise.

    Usage
    -----
    >>> is_unitary_matrix(np.eye(2))
    """
    if not is_square_matrix(matrix):
        return False

    matrix = np.conj(matrix.T).dot(matrix)
    return is_identity_matrix(matrix, ignore_phase=False, rtol=rtol, atol=atol)

def is_hermitian_matrix(
        matrix: NDArray[np.complex128],
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a Hermitian matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is a Hermitian matrix, False otherwise.

    Usage
    -----
    >>> is_hermitian_matrix(np.eye(2))
    """
    if not is_square_matrix(matrix):
        return False

    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    if matrix.ndim != 2:
        return False
    return np.allclose(matrix, np.conj(matrix.T), rtol=rtol, atol=atol)

def is_positive_semidefinite_matrix(
        matrix: NDArray[np.complex128],
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if a matrix is positive semidefinite.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is positive semidefinite, False otherwise.

    Usage
    -----
    >>> is_positive_semidefinite_matrix(np.eye(2))
    """
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    if not is_hermitian_matrix(matrix, rtol=rtol, atol=atol):
        return False

    # Check eigenvalues are all positive
    vals = np.linalg.eigvalsh(matrix)
    for v in vals:
        if v < -atol:
            return False
    return True

def is_isometry(
        matrix: NDArray[np.complex128],
        rtol: float=RTOL_DEFAULT,
        atol: float=ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is an isometry.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is an isometry, False otherwise.

    Usage
    -----
    >>> is_isometry(np.eye(2))
    """
    identity = np.eye(matrix.shape[1])
    matrix = np.conj(matrix.T).dot(matrix)
    return np.allclose(matrix, identity, rtol=rtol, atol=atol)