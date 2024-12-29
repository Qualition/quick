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

""" Weyl decomposition for two qubit unitary decomposition.
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
import itertools
import numpy as np
from numpy.typing import NDArray
import scipy.linalg # type: ignore

# Constants
B_NONNORAMLIZED = np.array([
    [1, 1j, 0, 0],
    [0, 0, 1j, 1],
    [0, 0, 1j, -1],
    [1, -1j, 0, 0]
], dtype=complex)

B_NONNORAMLIZED_DAGGER = 0.5 * B_NONNORAMLIZED.conj().T

# Pauli matrices in magic basis
_ipx = np.array([
    [0, 1j],
    [1j, 0]
], dtype=complex)

_ipy = np.array([
    [0, 1],
    [-1, 0]
], dtype=complex)

_ipz = np.array([
    [1j, 0],
    [0, -1j]
], dtype=complex)


def transform_to_magic_basis(
        U: NDArray[np.complex128],
        reverse: bool=False
    ) -> NDArray[np.complex128]:
    """ Transform the 4-by-4 matrix ``U`` into the magic basis.

    Notes
    -----
    This method internally uses non-normalized versions of the basis
    to minimize the floating-point errors that arise during the
    transformation.

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
        return B_NONNORAMLIZED_DAGGER @ U @ B_NONNORAMLIZED
    return B_NONNORAMLIZED @ U @ B_NONNORAMLIZED_DAGGER

def weyl_coordinates(U: NDArray[np.complex128]) -> NDArray[np.float64]:
    """ Calculate the Weyl coordinates for a given two-qubit unitary matrix.

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        The input 4-by-4 unitary matrix.

    Returns
    -------
    NDArray[np.float64]
        The array of the 3 Weyl coordinates.

    Usage
    -----
    >>> weyl_coordinates = weyl_coordinates(np.eye(4))
    """
    pi2 = np.pi / 2
    pi4 = np.pi / 4

    U = U / scipy.linalg.det(U) ** (0.25)
    Up = transform_to_magic_basis(U, reverse=True)

    # We only need the eigenvalues of `M2 = Up.T @ Up` here, not the full diagonalization
    D = scipy.linalg.eigvals(Up.T @ Up)

    d = -np.angle(D) / 2
    d[3] = -d[0] - d[1] - d[2]
    weyl_coordinates = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)

    # Reorder the eigenvalues to get in the Weyl chamber
    weyl_coordinates_temp = np.mod(weyl_coordinates, pi2)
    np.minimum(weyl_coordinates_temp, pi2 - weyl_coordinates_temp, weyl_coordinates_temp)
    order = np.argsort(weyl_coordinates_temp)[[1, 2, 0]]
    weyl_coordinates = weyl_coordinates[order]
    d[:3] = d[order]

    # Flip into Weyl chamber
    if weyl_coordinates[0] > pi2:
        weyl_coordinates[0] -= 3 * pi2
    if weyl_coordinates[1] > pi2:
        weyl_coordinates[1] -= 3 * pi2
    conjs = 0
    if weyl_coordinates[0] > pi4:
        weyl_coordinates[0] = pi2 - weyl_coordinates[0]
        conjs += 1
    if weyl_coordinates[1] > pi4:
        weyl_coordinates[1] = pi2 - weyl_coordinates[1]
        conjs += 1
    if weyl_coordinates[2] > pi2:
        weyl_coordinates[2] -= 3 * pi2
    if conjs == 1:
        weyl_coordinates[2] = pi2 - weyl_coordinates[2]
    if weyl_coordinates[2] > pi4:
        weyl_coordinates[2] -= pi2

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
    list[list[int]]
        The list of groups of indices of degenerate eigenvalues.

    Usage
    -----
    >>> groups = partition_eigenvalues(np.array([1, 1, 2, 2, 3, 3]))
    """
    groups: list[list[int]] = []
    for i, eigenvalue in enumerate(eigenvalues):
        for group in groups:
            for other in group:
                if abs(eigenvalue - eigenvalues[other]) > atol:
                    break
            else:
                group.append(i)
                break
        else:
            groups.append([i])
    return groups

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
    return (vector[index] / absolute[index]).conjugate() * vector

def diagonalize_unitary_complex_symmetric(
        U: NDArray[np.complex128],
        atol=1e-13
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """ Diagonalize the unitary complex-symmetric ``U`` with a complex
    diagonal matrix and a real-symmetric unitary matrix (in SO(4)).

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        The input unitary complex-symmetric matrix.
    `atol` : float, optional, default=1e-13
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
    # the determinant conditions will not be satisfied
    eigenvalues, eigenvectors = scipy.linalg.eig(U) # type: ignore
    eigenvalues /= np.abs(eigenvalues)

    # First find the degenerate subspaces, in order of dimension
    spaces = sorted(partition_eigenvalues(eigenvalues, atol=atol), key=len) # type: ignore

    if len(spaces) == 1:
        return eigenvalues, np.eye(4).astype(complex) # type: ignore

    out_vectors = np.empty((4, 4), dtype=np.float64)
    n_done = 0

    while n_done < 4 and len(spaces[n_done]) == 1:
        # 1D spaces must be only a global phase away from being real
        out_vectors[:, n_done] = remove_global_phase(eigenvectors[:, spaces[n_done][0]]).real # type: ignore
        n_done += 1

    if n_done == 0:
        # Two 2D spaces. This is the hardest case, because we might not have even one real vector
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

    # There can be at most one eigenspace not yet made real.  Since the whole vector basis is
    # orthogonal, the remaining space is equivalent to the null space of what we've got so far
    if n_done < 4:
        out_vectors[:, n_done:] = scipy.linalg.null_space(out_vectors[:, :n_done].T)

    # We assigned in space-dimension order, so we have to permute back to input order
    permutation = [None] * 4
    for i, x in enumerate(itertools.chain(*spaces)):
        permutation[x] = i # type: ignore
    out_vectors = out_vectors[:, permutation] # type: ignore

    # One extra orthogonalization to improve the overall tolerance, and ensure our minor floating
    # point twiddles haven't let us stray from normalisation
    out_vectors, _ = scipy.linalg.qr(out_vectors) # type: ignore

    return eigenvalues, out_vectors # type: ignore

def decompose_two_qubit_product_gate(
        special_unitary_matrix: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    """ Decompose U = Ul⊗Ur where U in SU(4), and Ul, Ur in SU(2).
    Throws ValueError if this isn't possible.

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
    detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        R = special_unitary_matrix[2:, :2].copy()
        detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        raise ValueError(f"The determinant of the right component must be more than 0.1. Received {detR}.")
    R /= np.sqrt(detR)

    # Extract the left component
    temp = np.kron(np.eye(2), R.T.conj())
    temp = special_unitary_matrix.dot(temp)
    L = temp[::2, ::2]
    detL = L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0]
    if abs(detL) < 0.9:
        raise ValueError(f"The determinant of the left component must be more than 0.9. Received {detL}.")
    L /= np.sqrt(detL)
    phase = cmath.phase(detL) / 2

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
        """ Initialize a `qickit.synthesis.gate_decompositions.two_qubit_decomposition.weyl.
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
        pi = np.pi
        pi2 = np.pi / 2
        pi4 = np.pi / 4

        # Make U be in SU(4)
        U = np.array(unitary_matrix, dtype=complex, copy=True)
        detU = scipy.linalg.det(U)
        U *= detU ** (-0.25)
        global_phase = cmath.phase(detU) / 4

        Up = transform_to_magic_basis(U.astype(complex), reverse=True)
        M2 = Up.T.dot(Up)
        D, P = diagonalize_unitary_complex_symmetric(M2)
        d = -np.angle(D) / 2
        d[3] = -d[0] - d[1] - d[2]
        weyl_coordinates = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)

        # Reorder the eigenvalues to get in the Weyl chamber
        weyl_coordinates_temp = np.mod(weyl_coordinates, pi2)
        np.minimum(weyl_coordinates_temp, pi2 - weyl_coordinates_temp, weyl_coordinates_temp)
        order = np.argsort(weyl_coordinates_temp)[[1, 2, 0]]
        weyl_coordinates = weyl_coordinates[order]
        d[:3] = d[order]
        P[:, :3] = P[:, order]

        # Fix the sign of P to be in SO(4)
        if np.real(scipy.linalg.det(P)) < 0:
            P[:, -1] = -P[:, -1]

        # Find K1, K2 so that U = K1.A.K2, with K being product of single-qubit unitaries
        K1 = transform_to_magic_basis(Up @ P @ np.diag(np.exp(1j * d)))
        K2 = transform_to_magic_basis(P.T)

        K1l, K1r, phase_l = decompose_two_qubit_product_gate(K1)
        K2l, K2r, phase_r = decompose_two_qubit_product_gate(K2)
        global_phase += phase_l + phase_r

        K1l = K1l.copy()

        # Flip into Weyl chamber
        if weyl_coordinates[0] > pi2:
            weyl_coordinates[0] -= 3 * pi2
            K1l = K1l.dot(_ipy)
            K1r = K1r.dot(_ipy)
            global_phase += pi2
        if weyl_coordinates[1] > pi2:
            weyl_coordinates[1] -= 3 * pi2
            K1l = K1l.dot(_ipx)
            K1r = K1r.dot(_ipx)
            global_phase += pi2
        conjs = 0
        if weyl_coordinates[0] > pi4:
            weyl_coordinates[0] = pi2 - weyl_coordinates[0]
            K1l = K1l.dot(_ipy)
            K2r = _ipy.dot(K2r)
            conjs += 1
            global_phase -= pi2
        if weyl_coordinates[1] > pi4:
            weyl_coordinates[1] = pi2 - weyl_coordinates[1]
            K1l = K1l.dot(_ipx)
            K2r = _ipx.dot(K2r)
            conjs += 1
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if weyl_coordinates[2] > pi2:
            weyl_coordinates[2] -= 3 * pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase += pi2
            if conjs == 1:
                global_phase -= pi
        if conjs == 1:
            weyl_coordinates[2] = pi2 - weyl_coordinates[2]
            K1l = K1l.dot(_ipz)
            K2r = _ipz.dot(K2r)
            global_phase += pi2
        if weyl_coordinates[2] > pi4:
            weyl_coordinates[2] -= pi2
            K1l = K1l.dot(_ipz)
            K1r = K1r.dot(_ipz)
            global_phase -= pi2

        a, b, c = weyl_coordinates[1], weyl_coordinates[0], weyl_coordinates[2]

        return a, b, c, K1l, K1r, K2l, K2r, global_phase