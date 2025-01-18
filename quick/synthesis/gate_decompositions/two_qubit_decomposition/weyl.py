# Copyright 2023-2024 Qualition Computing LLC.
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

""" Weyl decomposition for two qubit unitary decomposition.

This implementation is based on Jake Lishman's implementation for qiskit-terra:
https://github.com/jakelishman/qiskit-terra/tree/storage/deterministic-weyl-decomposition
"""

from __future__ import annotations

__all__ = [
    "transform_to_magic_basis",
    "weyl_coordinates",
    "partition_eigenvalues",
    "remove_global_phase",
    "diagonalize_unitary_complex_symmetric",
    "decompose_two_qubit_product_gate",
    "TwoQubitWeylDecomposition"
]

import cmath
from collections import defaultdict
import itertools
import numpy as np
from numpy.typing import NDArray
import scipy.linalg # type: ignore

""" Define the M matrix from section III to
tranform the unitary matrix into the magic basis:
https://arxiv.org/pdf/quant-ph/0308006

The basis M and its adjoint are stored individually unnormalized,
but such that their matrix multiplication is still the identity
This is because they are only used in unitary transformations
(so it's safe to do so), and `sqrt(0.5)` is not exactly representable
in floating point.

Doing it this way means that every element of the matrix is stored exactly
correctly, and the multiplication is exactly the identity rather than
differing by 1ULP.
"""
M_UNNORMALIZED = np.array([
    [1, 1j, 0, 0],
    [0, 0, 1j, 1],
    [0, 0, 1j, -1],
    [1, -1j, 0, 0]
], dtype=complex)

M_UNNORMALIZED_DAGGER = 0.5 * M_UNNORMALIZED.conj().T

# Pauli matrices in magic basis
X_MAGIC_BASIS = np.array([
    [0, 1j],
    [1j, 0]
], dtype=complex)

Y_MAGIC_BASIS = np.array([
    [0, 1],
    [-1, 0]
], dtype=complex)

Z_MAGIC_BASIS = np.array([
    [1j, 0],
    [0, -1j]
], dtype=complex)

# Constants
PI = np.pi
PI_DOUBLE = 2 * PI
PI2 = PI / 2
PI4 = PI / 4


def transform_to_magic_basis(
        U: NDArray[np.complex128],
        reverse: bool=False
    ) -> NDArray[np.complex128]:
    """ Transform the 4x4 matrix `U` into the magic basis.

    Notes
    -----
    This method internally uses non-normalized versions of the basis
    to minimize (but not eliminate) the floating-point errors that arise
    during the transformation.

    This implementation is based on the following paper:
    [1] Vatan, Williams.
    Optimal Quantum Circuits for General Two-Qubit Gates (2004).
    https://arxiv.org/abs/quant-ph/0308006

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        The input 4-by-4 matrix to be transformed.
    `reverse` : bool, optional, default=False
        If True, the transformation is done in the reverse direction.

    Returns
    -------
    NDArray[np.complex128]
        The transformed matrix in the magic basis.

    Usage
    -----
    >>> U_magic = transform_to_magic_basis(np.eye(4))
    """
    if reverse:
        return M_UNNORMALIZED_DAGGER @ U @ M_UNNORMALIZED
    return M_UNNORMALIZED @ U @ M_UNNORMALIZED_DAGGER

def weyl_coordinates(U: NDArray[np.complex128]) -> NDArray[np.float64]:
    """ Calculate the Weyl coordinates for a given two-qubit unitary matrix.
    This is used for unit-testing the Weyl decomposition.

    Notes
    -----
    This implementation is based on the following paper:
    [1] Cross, Bishop, Sheldon, Nation, Gambetta.
    Validating quantum computers using randomized model circuits (2019).
    https://arxiv.org/pdf/1811.12926

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        The input 4x4 unitary matrix.

    Returns
    -------
    NDArray[np.float64]
        The array of the 3 Weyl coordinates.

    Usage
    -----
    >>> weyl_coordinates = weyl_coordinates(np.eye(4))
    """
    U /= scipy.linalg.det(U) ** (0.25)
    U_magic_basis = transform_to_magic_basis(U, reverse=True)

    # We only need the eigenvalues of `M2 = Up.T @ Up` here, not the full diagonalization
    D = scipy.linalg.eigvals(U_magic_basis.T @ U_magic_basis)

    d = -np.angle(D) / 2
    d[3] = -d[0] - d[1] - d[2]
    weyl_coordinates = np.mod((d[:3] + d[3]) / 2, PI_DOUBLE)

    # Reorder the eigenvalues to get in the Weyl chamber
    weyl_coordinates_temp = np.mod(weyl_coordinates, PI2)
    np.minimum(weyl_coordinates_temp, PI2 - weyl_coordinates_temp, weyl_coordinates_temp)
    order = np.argsort(weyl_coordinates_temp)[[1, 2, 0]]
    weyl_coordinates = weyl_coordinates[order]
    d[:3] = d[order]

    # Flip into Weyl chamber
    if weyl_coordinates[0] > PI2:
        weyl_coordinates[0] -= 3 * PI2
    if weyl_coordinates[1] > PI2:
        weyl_coordinates[1] -= 3 * PI2
    conjs = 0
    if weyl_coordinates[0] > PI4:
        weyl_coordinates[0] = PI2 - weyl_coordinates[0]
        conjs += 1
    if weyl_coordinates[1] > PI4:
        weyl_coordinates[1] = PI2 - weyl_coordinates[1]
        conjs += 1
    if weyl_coordinates[2] > PI2:
        weyl_coordinates[2] -= 3 * PI2
    if conjs == 1:
        weyl_coordinates[2] = PI2 - weyl_coordinates[2]
    if weyl_coordinates[2] > PI4:
        weyl_coordinates[2] -= PI2

    return weyl_coordinates[[1, 0, 2]]

def partition_eigenvalues(
        eigenvalues: NDArray[np.complex128],
        atol: float=1e-13
    ) -> list[list[int]]:
    """ Group the indices of degenerate eigenvalues.

    Parameters
    ----------
    `eigenvalues` : NDArray[np.complex128]
        The array of eigenvalues.
    `atol` : float, optional, default=1e-13
        The absolute tolerance for grouping the eigenvalues.

    Returns
    -------
    `groups` : list[list[int]]
        The list of groups of indices of degenerate eigenvalues.

    Usage
    -----
    >>> groups = partition_eigenvalues(np.array([1, 1, 2, 2, 3, 3]))
    """
    groups: defaultdict = defaultdict(list)
    for i, eigenvalue in enumerate(eigenvalues):
        for key in groups:
            if abs(eigenvalue - key) <= atol:
                groups[key].append(i)
                break
        else:
            groups[eigenvalue].append(i)
    return list(groups.values())

def remove_global_phase(
        vector: NDArray[np.complex128],
        index=None
    ) -> NDArray[np.complex128]:
    """ Rotate the vector by the negative argument of the largest
    absolute element.

    Parameters
    ----------
    `vector` : NDArray[np.complex128]
        The input vector.
    `index` : int, optional, default=None
        The index of the element to be considered for the rotation.

    Returns
    -------
    NDArray[np.complex128]
        The rotated vector.

    Usage
    -----
    >>> phaseless_vector = remove_global_phase(np.array([1, 1j, 0, 0]))
    """
    absolute = np.abs(vector)
    index = np.argmax(absolute) if index is None else index
    return (vector[index] / absolute[index]).conj() * vector

def diagonalize_unitary_complex_symmetric(
        U: NDArray[np.complex128],
        atol=1e-19
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """ Diagonalize the unitary complex-symmetric `U` with a complex
    diagonal matrix and a real-symmetric unitary matrix (in SO(4)).

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        The input unitary complex-symmetric matrix.
    `atol` : float, optional, default=1e-19
        The absolute tolerance for the diagonalization.

    Returns
    -------
    `eigenvalues` : NDArray[np.complex128]
        The array of eigenvalues.
    `out_vectors` : NDArray[np.complex128]
        The array of eigenvectors.

    Usage
    -----
    >>> eigenvalues, out_vectors = diagonalize_unitary_complex_symmetric(np.eye(4))
    """
    # Use `scipy.linalg.eig` to get the eigenvalues and eigenvectors
    # If we use `scipy.linalg.eigh`, the decomposition will fail given
    # the determinant conditions will not be satisfied as U is not necceasrily
    # symmetric
    # Additionally, `scipy.linalg.eig()` can simultaneously compute both
    # left and right eigenvectors, while `numpy.linalg.eig()` can only compute
    # right eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eig(U) # type: ignore
    eigenvalues /= np.abs(eigenvalues)

    # First find the degenerate subspaces, in order of dimension
    spaces = sorted(partition_eigenvalues(eigenvalues, atol=atol), key=len) # type: ignore

    # If there are no degenerate subspaces, we return the eigenvalues and identity matrix
    # as the eigenvectors
    if len(spaces) == 1:
        return eigenvalues, np.eye(4).astype(complex) # type: ignore

    out_vectors = np.empty((4, 4), dtype=np.float64)
    n_done = 0

    while n_done < 4 and len(spaces[n_done]) == 1:
        # 1D spaces must be only a global phase away from being real
        out_vectors[:, n_done] = remove_global_phase(eigenvectors[:, spaces[n_done][0]]).real # type: ignore
        n_done += 1

    if n_done == 0:
        # Two 2D spaces
        # This is the hardest case, because there might not have even one real vector
        a, b = eigenvectors[:, spaces[0]].T
        b_zeros = np.abs(b) <= atol
        if np.any(np.abs(a[b_zeros]) > atol):
            # Make `a` real where `b` has zeros.
            a = remove_global_phase(a, index=np.argmax(np.where(b_zeros, np.abs(a), 0)))
        if np.max(np.abs(a.imag)) <= atol:
            # `a` is already all real
            pass
        else:
            # We have to solve `(b.imag, b.real) @ (re, im).T = a.imag` for `re`
            # and `im`, which is overspecified
            multiplier, *_ = scipy.linalg.lstsq(np.transpose([b.imag, b.real]), a.imag)
            a = a - complex(*multiplier) * b
        a = a.real / scipy.linalg.norm(a.real)
        b = remove_global_phase(b - (a @ b) * a)
        out_vectors[:, :2] = np.transpose([a, b.real])
        n_done = 2

    # There can be at most one eigenspace not yet made real
    # Since the whole vector basis is orthogonal the remaining
    # space is equivalent to the null space of what we've got
    # so far
    if n_done < 4:
        out_vectors[:, n_done:] = scipy.linalg.null_space(out_vectors[:, :n_done].T)

    # We assigned in space-dimension order so we have to permute back to input order
    permutation = [None] * 4
    for i, x in enumerate(itertools.chain(*spaces)):
        permutation[x] = i # type: ignore
    out_vectors = out_vectors[:, permutation] # type: ignore

    # Perform an additional orthogonalization to improve overall tolerance and ensure
    # that minor floating point adjustments have not affected normalization
    out_vectors, _ = scipy.linalg.qr(out_vectors) # type: ignore

    return eigenvalues, out_vectors # type: ignore

def decompose_two_qubit_product_gate(
        special_unitary_matrix: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    """ Decompose $U = U_l \otimes U_r$ where $U$ in SU(4), and $U_l$, $U_r$ in SU(2).
    Throws ValueError if this decomposition isn't possible.

    Parameters
    ----------
    `special_unitary_matrix` : NDArray[np.complex128]
        The input special unitary matrix.

    Returns
    -------
    `L` : NDArray[np.complex128]
        The left component.
    `R` : NDArray[np.complex128]
        The right component.
    `phase` : float
        The phase.

    Raises
    ------
    ValueError
        - If the decomposition is not possible.

    Usage
    -----
    >>> L, R, phase = decompose_two_qubit_product_gate(np.eye(4))
    """
    special_unitary_matrix = np.asarray(special_unitary_matrix, dtype=complex)

    # Extract the right component
    R = special_unitary_matrix[:2, :2].copy()
    R_det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]

    if abs(R_det) < 0.1:
        R = special_unitary_matrix[2:, :2].copy()
        R_det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]

    if abs(R_det) < 0.1:
        raise ValueError(
            "The determinant of the right component must be more than 0.1. "
            f"Received {R_det}."
        )

    R /= np.sqrt(R_det)

    # Extract the left component
    temp = np.kron(np.eye(2), R.T.conj())
    temp = special_unitary_matrix.dot(temp)
    L = temp[::2, ::2]
    L_det = L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0]

    if abs(L_det) < 0.9:
        raise ValueError(
            "The determinant of the left component must be more than 0.9. "
            f"Received {L_det}."
        )

    L /= np.sqrt(L_det)
    phase = cmath.phase(L_det) / 2

    temp = np.kron(L, R)
    deviation = abs(abs(temp.conj().T.dot(special_unitary_matrix).trace()) - 4)
    if deviation > 1e-13:
        raise ValueError(f"Decomposition failed. Deviation: {deviation}.")

    return L, R, phase


class TwoQubitWeylDecomposition:
    """ Decompose a two-qubit unitary matrix into the Weyl coordinates and
    the product of two single-qubit unitaries.

    Parameters
    ----------
    `unitary_matrix` : NDArray[np.complex128]
        The input 4-by-4 unitary matrix.

    Attributes
    ----------
    `a` : np.float64
        The first Weyl coordinate.
    `b` : np.float64
        The second Weyl coordinate.
    `c` : np.float64
        The third Weyl coordinate.
    `K1l` : NDArray[np.complex128]
        The left component of the first single-qubit unitary.
    `K1r` : NDArray[np.complex128]
        The right component of the first single-qubit unitary.
    `K2l` : NDArray[np.complex128]
        The left component of the second single-qubit unitary.
    `K2r` : NDArray[np.complex128]
        The right component of the second single-qubit unitary.
    `global_phase` : float
        The global phase.
    """
    def __init__(self, unitary_matrix: NDArray[np.complex128]) -> None:
        """ Initialize a `quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl.
        TwoQubitWeylDecomposition` instance.
        """
        self.a, self.b, self.c, self.K1l, self.K1r, self.K2l, self.K2r, self.global_phase = self.decompose_unitary(
            unitary_matrix
        )

    @staticmethod
    def decompose_unitary(unitary_matrix: NDArray[np.complex128]) -> tuple[
            np.float64,
            np.float64,
            np.float64,
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            float
        ]:
        """ Decompose a two-qubit unitary matrix into the Weyl coordinates and the product of two single-qubit unitaries.

        Parameters
        ----------
        `unitary_matrix` : NDArray[np.complex128]
            The input 4-by-4 unitary matrix.

        Returns
        -------
        `a` : np.float64
            The first Weyl coordinate.
        `b` : np.float64
            The second Weyl coordinate.
        `c` : np.float64
            The third Weyl coordinate.
        `K1l` : NDArray[np.complex128]
            The left component of the first single-qubit unitary.
        `K1r` : NDArray[np.complex128]
            The right component of the first single-qubit unitary.
        `K2l` : NDArray[np.complex128]
            The left component of the second single-qubit unitary.
        `K2r` : NDArray[np.complex128]
            The right component of the second single-qubit unitary.
        `global_phase` : float
            The global phase.

        Usage
        -----
        >>> a, b, c, K1l, K1r, K2l, K2r, global_phase = TwoQubitWeylDecomposition.decompose_unitary(np.eye(4))
        """
        # Make U be in SU(4)
        U = np.array(unitary_matrix, dtype=complex, copy=True)
        U_det = scipy.linalg.det(U)
        U *= U_det ** (-0.25)
        global_phase = cmath.phase(U_det) / 4

        U_magic_basis = transform_to_magic_basis(U.astype(complex), reverse=True)
        M2 = U_magic_basis.T.dot(U_magic_basis)

        # NOTE: There is a floating point error in this implementation
        # for certain U, which depends on OS and Python version
        # This causes the numpy.linalg.eig() to produce different results
        # for the same input matrix, leading to a decomposition failure
        # To contribute to this issue, please refer to:
        # https://github.com/Qualition/quick/issues/11

        # NOTE: Alternatively, you may propose an entirely new implementation
        # so that we can replace this two qubit decomposition implementation
        # with a more robust one that doesn't have floating point errors
        # To contribute to this feature request, please refer to:
        # https://github.com/Qualition/quick/issues/14

        D, P = diagonalize_unitary_complex_symmetric(M2)
        d = -np.angle(D) / 2
        d[3] = -d[0] - d[1] - d[2]
        weyl_coordinates = np.mod((d[:3] + d[3]) / 2, PI_DOUBLE)

        # Reorder the eigenvalues to get in the Weyl chamber
        weyl_coordinates_temp = np.mod(weyl_coordinates, PI2)
        np.minimum(weyl_coordinates_temp, PI2 - weyl_coordinates_temp, weyl_coordinates_temp)
        order = np.argsort(weyl_coordinates_temp)[[1, 2, 0]]
        weyl_coordinates = weyl_coordinates[order]
        d[:3] = d[order]
        P[:, :3] = P[:, order]

        # Fix the sign of P to be in SO(4)
        if np.real(scipy.linalg.det(P)) < 0:
            P[:, -1] = -P[:, -1]

        # Find K1, K2 so that U = K1.A.K2, with K being product of single-qubit unitaries
        K1 = transform_to_magic_basis(U_magic_basis @ P @ np.diag(np.exp(1j * d)))
        K2 = transform_to_magic_basis(P.T)

        K1l, K1r, phase_l = decompose_two_qubit_product_gate(K1)
        K2l, K2r, phase_r = decompose_two_qubit_product_gate(K2)
        global_phase += phase_l + phase_r

        K1l = K1l.copy()

        # Flip into Weyl chamber
        if weyl_coordinates[0] > PI2:
            weyl_coordinates[0] -= 3 * PI2
            K1l = K1l.dot(Y_MAGIC_BASIS)
            K1r = K1r.dot(Y_MAGIC_BASIS)
            global_phase += PI2
        if weyl_coordinates[1] > PI2:
            weyl_coordinates[1] -= 3 * PI2
            K1l = K1l.dot(X_MAGIC_BASIS)
            K1r = K1r.dot(X_MAGIC_BASIS)
            global_phase += PI2
        conjs = 0
        if weyl_coordinates[0] > PI4:
            weyl_coordinates[0] = PI2 - weyl_coordinates[0]
            K1l = K1l.dot(Y_MAGIC_BASIS)
            K2r = Y_MAGIC_BASIS.dot(K2r)
            conjs += 1
            global_phase -= PI2
        if weyl_coordinates[1] > PI4:
            weyl_coordinates[1] = PI2 - weyl_coordinates[1]
            K1l = K1l.dot(X_MAGIC_BASIS)
            K2r = X_MAGIC_BASIS.dot(K2r)
            conjs += 1
            global_phase += PI2
            if conjs == 1:
                global_phase -= PI
        if weyl_coordinates[2] > PI2:
            weyl_coordinates[2] -= 3 * PI2
            K1l = K1l.dot(Z_MAGIC_BASIS)
            K1r = K1r.dot(Z_MAGIC_BASIS)
            global_phase += PI2
            if conjs == 1:
                global_phase -= PI
        if conjs == 1:
            weyl_coordinates[2] = PI2 - weyl_coordinates[2]
            K1l = K1l.dot(Z_MAGIC_BASIS)
            K2r = Z_MAGIC_BASIS.dot(K2r)
            global_phase += PI2
        if weyl_coordinates[2] > PI4:
            weyl_coordinates[2] -= PI2
            K1l = K1l.dot(Z_MAGIC_BASIS)
            K1r = K1r.dot(Z_MAGIC_BASIS)
            global_phase -= PI2

        a, b, c = weyl_coordinates[1], weyl_coordinates[0], weyl_coordinates[2]

        return a, b, c, K1l, K1r, K2l, K2r, global_phase