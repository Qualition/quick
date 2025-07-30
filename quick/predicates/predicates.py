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

""" Predicates module.
"""

from __future__ import annotations

__all__ = [
    "is_statevector",
    "is_square_matrix",
    "is_diagonal_matrix",
    "is_symmetric_matrix",
    "is_identity_matrix",
    "is_unitary_matrix",
    "is_hermitian_matrix",
    "is_positive_semidefinite_matrix",
    "is_isometry",
    "is_density_matrix"
]

import numpy as np
from numpy.typing import NDArray
import math

ATOL_DEFAULT = 1e-8
RTOL_DEFAULT = 1e-5


def _is_power(
        base: int,
        number: int
    ) -> bool:
    """ Test if a number is a power of another number.

    Parameters
    ----------
    `base` : int
        The base number.
    `number` : int
        The number to check.

    Returns
    -------
    bool
        True if the number is a power of the base, False otherwise.
    """
    result = math.log(number) / math.log(base)
    return result == math.floor(result)

def is_statevector(
        statevector: NDArray[np.complex128],
        system_size: int = 2,
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a statevector.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The input statevector.
    `system_size` : int, optional, default=2
        The size of the quantum memory. If the size is 2, then the
        system uses qubits. If the size is 3, then the system uses qutrits,
        and so on.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the array is a statevector, False otherwise.

    Raises
    ------
    ValueError
        - If the system size is less than 2.

    Usage
    -----
    >>> is_statevector(np.array([1, 0]))
    """
    if system_size < 2:
        raise ValueError("System size must be greater than or equal to 2.")

    if not _is_power(system_size, len(statevector)):
        return False

    if statevector.ndim == 2:
        if statevector.shape[1] == 1:
            statevector = statevector.ravel()

    return (
        bool(np.isclose(np.linalg.norm(statevector), 1.0, rtol=rtol, atol=atol))
        and statevector.ndim == 1
        and len(statevector) > 1
    )

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
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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

    return np.allclose(matrix, np.diag(np.diagonal(matrix)), rtol=rtol, atol=atol)

def is_symmetric_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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

    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

def is_identity_matrix(
        matrix: NDArray[np.complex128],
        ignore_phase: bool = False,
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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

    if ignore_phase:
        # If the matrix is equal to an identity up to a phase, we can
        # remove the phase by multiplying each entry by the complex
        # conjugate of the phase of the [0, 0] entry.
        theta = np.angle(matrix[0, 0])
        matrix = np.exp(-1j * theta) * matrix

    identity = np.eye(len(matrix))
    return np.allclose(matrix, identity, rtol=rtol, atol=atol)

def is_unitary_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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

    matrix = matrix.conj().T @ matrix
    return is_identity_matrix(matrix, ignore_phase=False, rtol=rtol, atol=atol)

def is_hermitian_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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

    return np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol)

def is_positive_semidefinite_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
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
    if matrix.ndim != 2:
        return False

    identity = np.eye(matrix.shape[1])
    matrix = matrix.conj().T @ matrix
    return np.allclose(matrix, identity, rtol=rtol, atol=atol)

def is_density_matrix(
        rho: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a density matrix.

    Parameters
    ----------
    `rho` : NDArray[np.complex128]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is a density matrix, False otherwise.

    Usage
    -----
    >>> is_density_matrix(np.eye(2))
    """
    if not (
        is_hermitian_matrix(rho, rtol=rtol, atol=atol)
        and is_positive_semidefinite_matrix(rho, rtol=rtol, atol=atol)
        and np.isclose(np.trace(rho), 1.0, rtol=rtol, atol=atol)
    ):
        return False

    return True