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
    "is_power",
    "is_normalized",
    "is_statevector",
    "is_square_matrix",
    "is_orthogonal_matrix",
    "is_real_matrix",
    "is_special_matrix",
    "is_special_orthogonal_matrix",
    "is_special_unitary_matrix",
    "is_diagonal_matrix",
    "is_symmetric_matrix",
    "is_identity_matrix",
    "is_unitary_matrix",
    "is_hermitian_matrix",
    "is_positive_semidefinite_matrix",
    "is_isometry",
    "is_density_matrix",
    "is_product_matrix",
    "is_locally_equivalent",
    "is_supercontrolled"
]

import numpy as np
from numpy.typing import NDArray
import math

ATOL_DEFAULT = 1e-8
RTOL_DEFAULT = 1e-5


def is_power(
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
    return bool(result == math.floor(result))

def is_normalized(
        statevector: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is normalized.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The input statevector.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the array is normalized, False otherwise.

    Usage
    -----
    >>> is_normalized(np.array([1, 0]))
    """
    return bool(np.isclose(np.linalg.norm(statevector), 1.0, rtol=rtol, atol=atol))

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

    if not is_power(system_size, len(statevector)):
        return False

    if statevector.ndim == 2:
        if statevector.shape[1] == 1:
            statevector = statevector.ravel()

    return bool(
        is_normalized(statevector, rtol=rtol, atol=atol)
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
    return bool(shape[0] == shape[1])

def is_orthogonal_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is an orthogonal matrix.

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
        True if the matrix is an orthogonal matrix, False otherwise.

    Usage
    -----
    >>> is_orthogonal_matrix(np.eye(2))
    """
    return bool(np.allclose(matrix.T, np.linalg.inv(matrix), rtol=rtol, atol=atol))

def is_real_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a real matrix.

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
        True if the matrix is a real matrix, False otherwise.

    Usage
    -----
    >>> is_real_matrix(np.eye(2))
    """
    return bool(np.allclose(matrix, matrix.real, rtol=rtol, atol=atol))

def is_special_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a special matrix (i.e., has determinant 1).

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
        True if the matrix is a special matrix, False otherwise.

    Usage
    -----
    >>> is_special_matrix(np.eye(2))
    """
    if not is_square_matrix(matrix):
        return False

    det = np.linalg.det(matrix)
    return bool(np.isclose(det, 1.0, rtol=rtol, atol=atol))

def is_special_orthogonal_matrix(
        matrix: NDArray[np.float64],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a special orthogonal matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.float64]
        The input matrix.
    `rtol` : float, optional, default=RTOL_DEFAULT
        The relative tolerance parameter.
    `atol` : float, optional, default=ATOL_DEFAULT
        The absolute tolerance parameter.

    Returns
    -------
    bool
        True if the matrix is a special orthogonal matrix, False otherwise.

    Usage
    -----
    >>> is_so(np.eye(2))
    """
    return bool(
        is_special_matrix(matrix.astype(complex), rtol=rtol, atol=atol)
        and is_orthogonal_matrix(matrix.astype(complex), rtol=rtol, atol=atol)
    )

def is_special_unitary_matrix(
        matrix: NDArray[np.complex128],
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT
    ) -> bool:
    """ Test if an array is a special unitary matrix.

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
        True if the matrix is a special unitary matrix, False otherwise.

    Usage
    -----
    >>> is_su(np.eye(2))
    """
    return bool(
        is_special_matrix(matrix, rtol=rtol, atol=atol)
        and is_unitary_matrix(matrix, rtol=rtol, atol=atol)
    )

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

    return bool(np.allclose(matrix, np.diag(np.diagonal(matrix)), rtol=rtol, atol=atol))

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

    return bool(np.allclose(matrix, matrix.T, rtol=rtol, atol=atol))

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
    return bool(np.allclose(matrix, identity, rtol=rtol, atol=atol))

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
    return bool(is_identity_matrix(matrix, ignore_phase=False, rtol=rtol, atol=atol))

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

    return bool(np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol))

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
    return bool(np.allclose(matrix, identity, rtol=rtol, atol=atol))

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
    if not bool(
        is_hermitian_matrix(rho, rtol=rtol, atol=atol)
        and is_positive_semidefinite_matrix(rho, rtol=rtol, atol=atol)
        and np.isclose(np.trace(rho), 1.0, rtol=rtol, atol=atol)
    ):
        return False

    return True

def is_product_matrix(matrix: NDArray[np.complex128]) -> bool:
    """ Test if a two-qubit unitary is a product matrix.

    A two-qubit gate is a product matrix if it can be expressed as
    the Kronecker product of two single-qubit gates.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input two-qubit unitary matrix.

    Returns
    -------
    bool
        True if the matrix is a product matrix, False otherwise.

    Raises
    ------
    ValueError
        - If the input matrix is not a two-qubit unitary.

    Usage
    -----
    >>> from quick.synthesis.gate_decompositions.two_qubit_decomposition import swap
    >>> is_product_matrix(swap())
    False
    """
    from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import weyl_coordinates

    if not is_unitary_matrix(matrix) or matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 unitary matrix.")

    x, y, z = weyl_coordinates(matrix)

    return bool(np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 0))

def is_locally_equivalent(
        matrix1: NDArray[np.complex128],
        matrix2: NDArray[np.complex128]
    ) -> bool:
    """ Test if two two-qubit unitaries are locally equivalent.
    Two two-qubit gates are locally equivalent if they differ only
    by single-qubit gates.

    Parameters
    `matrix1` : NDArray[np.complex128]
        The first input two-qubit unitary matrix.
    `matrix2` : NDArray[np.complex128]
        The second input two-qubit unitary matrix.

    Returns
    -------
    bool
        True if the matrices are locally equivalent, False otherwise.

    Raises
    ------
    ValueError
        - If either input matrix is not a two-qubit unitary.

    Usage
    -----
    >>> is_locally_equivalent(cx, cz)
    """
    from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import weyl_coordinates

    if not is_unitary_matrix(matrix1) or matrix1.shape != (4, 4):
        raise ValueError("First input matrix must be a 4x4 unitary matrix.")
    if not is_unitary_matrix(matrix2) or matrix2.shape != (4, 4):
        raise ValueError("Second input matrix must be a 4x4 unitary matrix.")

    x1, y1, z1 = weyl_coordinates(matrix1)
    x2, y2, z2 = weyl_coordinates(matrix2)

    return bool(np.isclose(x1, x2) and np.isclose(y1, y2) and np.isclose(z1, z2))

def is_supercontrolled(matrix: NDArray[np.complex128]) -> bool:
    """ Test if a two-qubit unitary is a supercontrolled gate.

    A two-qubit gate is supercontrolled if its Weyl coordinates
    are of the form (pi/4, alpha, 0) up to local equivalence.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input two-qubit unitary matrix.

    Returns
    -------
    bool
        True if the matrix is a supercontrolled gate, False otherwise.

    Raises
    ------
    ValueError
        - If the input matrix is not a two-qubit unitary.

    Usage
    -----
    >>> from quick.synthesis.gate_decompositions.two_qubit_decomposition import cnot
    >>> is_supercontrolled(cnot())
    True
    """
    from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import weyl_coordinates

    if not is_unitary_matrix(matrix) or matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 unitary matrix.")

    x, _, z = weyl_coordinates(matrix)

    return bool(np.isclose(x, np.pi / 4) and np.isclose(z, 0))