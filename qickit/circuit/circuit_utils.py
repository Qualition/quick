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

""" Helper functions for quantum circuits.
"""

from __future__ import annotations

__all__ = [
    "update_angles",
    "decompose_uc_rotations",
    "extract_rz",
    "det_one_qubit",
    "demultiplex_single_uc",
    "decompose_ucg_help",
    "get_ucg_diagonal",
    "simplify",
    "repetition_search",
    "repetition_verify"
]

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import eig
from math import pi

EPS = 1e-10
FRAC_1_SQRT_2 = 1 / np.sqrt(2)
RZ_PI2_11 = complex(FRAC_1_SQRT_2, -FRAC_1_SQRT_2)
RZ_PI2_00 = complex(FRAC_1_SQRT_2, FRAC_1_SQRT_2)
IM = 1j
C_ZERO = 0 + 0j


def update_angles(
        angle_1: float,
        angle_2: float
    ) -> tuple[float, float]:
    """ Calculate the new rotation angles for Uniformly Controlled Pauli Rotation
    gate according to Shende's decomposition.

    Parameters
    ----------
    `angle_1` : float
        The first rotation angle.
    `angle_2` : float
        The second rotation angle.

    Returns
    -------
    tuple[float, float]
        The new rotation angles.
    """
    return (angle_1 + angle_2) / 2.0, (angle_1 - angle_2) / 2.0

def decompose_uc_rotations(
        angles: NDArray[np.float64],
        start_index: int,
        end_index: int,
        reversed_dec: bool
    ) -> None:
    """ Calculate rotation angles for a Uniformly Controlled Pauli Rotation gate
    with a CX gate at the end of the circuit. The rotation angles of the rotation
    gates are stored in angles[start_index:end_index]. If reversed_dec == True,
    it decomposes the gate such that there is a CX gate at the start of the circuit
    (in fact, the circuit topology for the reversed decomposition is the reversed one
    of the original decomposition)

    Parameters
    ----------
    `angles` : NDArray[np.float64]
        The list of rotation angles.
    `start_index` : int
        The start index of the rotation angles.
    `end_index` : int
        The end index of the rotation angles.
    """
    interval_len_half = (end_index - start_index) // 2

    # Recursively decompose the first half of the interval
    for i in range(start_index, start_index + interval_len_half):
        if not reversed_dec:
            angles[i], angles[i + interval_len_half] = update_angles(
                angles[i], angles[i + interval_len_half]
            )
        else:
            angles[i + interval_len_half], angles[i] = update_angles(
                angles[i], angles[i + interval_len_half]
            )

    if interval_len_half <= 1:
        return

    # Decompose the second half of the interval
    else:
        decompose_uc_rotations(
            angles, start_index, start_index + interval_len_half, False
        )
        decompose_uc_rotations(
            angles, start_index + interval_len_half, end_index, True
        )

def extract_rz(
        phi_1: float,
        phi_2: float
    ) -> tuple[float, float]:
    """ Extract a Rz rotation (angle given by first output) such that
    exp(j*phase)*Rz(z_angle) is equal to the diagonal matrix with entires
    exp(1j*ph1) and exp(1j*ph2).

    Parameters
    ----------
    `phi_1` : float
        The first phase angle.
    `phi_2` : float
        The second phase angle.

    Returns
    -------
    tuple[float, float]
        The phase angle and the z angle.
    """
    phase = (phi_1 + phi_2) / 2.0
    z_angle = phi_2 - phi_1
    return phase, z_angle

def det_one_qubit(matrix: NDArray[np.complex128]) -> np.complex64:
    """ Calculate the determinant of a 2x2 matrix.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The input matrix.

    Returns
    -------
    np.complex64
        The determinant of the matrix.
    """
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def demultiplex_single_uc(
        a: NDArray[np.complex128],
        b: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """ Demultiplex a single uniformly controlled gate.

    Parameters
    ----------
    `a` : NDArray[np.complex128]
        The first single qubit gate.
    `b` : NDArray[np.complex128]
        The second single qubit gate.

    Returns
    -------
    `v` : NDArray[np.complex128]
        The first single qubit gate.
    `u` : NDArray[np.complex128]
        The second single qubit gate.
    `r` : NDArray[np.complex128]
        The rotation matrix.
    """
    # Hermitian conjugate of b
    x = a @ np.conj(b).T

    # Determinant and phase of x
    det_x = det_one_qubit(x) # type: ignore
    x11 = x[0, 0] / np.sqrt(det_x)
    phi = np.angle(det_x)

    # Compute the rotation matrix r
    r1 = np.exp(IM / 2 * (pi / 2 - phi / 2 - np.angle(x11)))
    r2 = np.exp(IM / 2 * (pi / 2 - phi / 2 + np.angle(x11) + pi))
    r = np.array([[r1, C_ZERO], [C_ZERO, r2]])

    # Eigen decomposition of r @ x @ r
    rxr = r @ x @ r
    eigvals, u = eig(rxr)

    # Put the eigenvalues into a diagonal form
    diag = np.diag(np.sqrt(eigvals))

    # Handle specific case where the eigenvalue is near -i
    if np.abs(diag[0, 0] + IM) < EPS:
        diag = np.flipud(diag)
        u = np.fliplr(u)

    # Calculate v based on the decomposition
    v = diag @ np.conj(u).T @ np.conj(r).T @ b

    return v, u, r

def decompose_ucg_help(
        sq_gates: list[NDArray[np.complex128]],
        num_qubits: int
    ) -> tuple[list[NDArray[np.complex128]], NDArray[np.complex128]]:
    """ Find the single qubit gate arising in the decomposition of Uniformly
    Controlled gates given in the paper by Bergholm et al.

    Notes
    -----
    This function is used to decompose uniformly controlled gates based on
    the paper by Bergholm et al.

    Bergholm, Vartiainen, Möttönen, Salomaa,
    Quantum circuits with uniformly controlled one-qubit gates (2005).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330

    Parameters
    ----------
    `sq_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.
    `num_qubits` : int
        The number of qubits.

    Returns
    -------
    `single_qubit_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.
    `diagonal` : NDArray[np.complex128]
        The diagonal matrix.
    """
    single_qubit_gates = [np.copy(gate) for gate in sq_gates]
    diagonal = np.ones(2**num_qubits, dtype=np.complex128)
    num_controls = num_qubits - 1

    for dec_step in range(num_controls):
        num_ucgs = 2**dec_step
        for ucg_index in range(num_ucgs):
            len_ucg = 2**(num_controls - dec_step)
            for i in range(len_ucg // 2):
                shift = ucg_index * len_ucg
                a = single_qubit_gates[shift + i]
                b = single_qubit_gates[shift + len_ucg // 2 + i]

                v, u, r = demultiplex_single_uc(a, b)

                # Replace the single qubit gates
                single_qubit_gates[shift + i] = v
                single_qubit_gates[shift + len_ucg // 2 + i] = u

                # Decompose D gates as described in the paper
                r_conj_t = np.conj(r).T
                if ucg_index < num_ucgs - 1:
                    k = shift + len_ucg + i
                    single_qubit_gates[k] = single_qubit_gates[k] @ r_conj_t
                    single_qubit_gates[k] *= RZ_PI2_00
                    k += len_ucg // 2
                    single_qubit_gates[k] = single_qubit_gates[k] @ r
                    single_qubit_gates[k] *= RZ_PI2_11
                else:
                    for ucg_index_2 in range(num_ucgs):
                        shift_2 = ucg_index_2 * len_ucg
                        k = 2 * (i + shift_2)
                        diagonal[k] *= r_conj_t[0, 0] * RZ_PI2_00
                        diagonal[k + 1] *= r_conj_t[1, 1] * RZ_PI2_00
                        k += len_ucg
                        diagonal[k] *= r[0, 0] * RZ_PI2_11
                        diagonal[k + 1] *= r[1, 1] * RZ_PI2_11

    return single_qubit_gates, diagonal

def get_ucg_diagonal(
        sq_gates: list[NDArray[np.complex128]],
        num_qubits: int,
        simplified_controls: set[int]
    ) -> NDArray[np.complex128]:
    """ Get the diagonal matrix arising in the decomposition of Uniformly

    Controlled gates given in the paper by Bergholm et al.

    Parameters
    ----------
    `sq_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.
    `num_qubits` : int
        The number of qubits.

    Returns
    -------
    NDArray[np.complex128]
        The diagonal matrix.
    """
    _, diagonal = decompose_ucg_help(sq_gates, num_qubits)
    if simplified_controls:
        q_controls = [num_qubits - i for i in simplified_controls]
        q_controls.reverse()
        for i in range(num_qubits):
            if i not in [0] + q_controls:
                d = 2**i
                new_diagonal = []
                n = len(diagonal)
                for j in range(n):
                    new_diagonal.append(diagonal[j])
                    if (j + 1) % d == 0:
                        new_diagonal.extend(diagonal[j + 1 - d : j + 1])
                diagonal = np.array(new_diagonal)
    return diagonal

def simplify(
        gate_list: list[NDArray[np.complex128]],
        num_controls: int
    ) -> tuple[set[int], list[NDArray[np.complex128]]]:
    """ Perform the simplification given in the paper by de Carvalho et al.

    Parameters
    ----------
    `gate_list` : list[NDArray[np.complex128]]
        The list of gates.
    `num_controls` : int
        The number of controls.

    Returns
    -------
    `new_controls` : set[int]
        The new set of controls.
    `new_mux` : list[NDArray[np.complex128]]
        The new list of gates.
    """
    c: set[int] = set()
    nc: set[int] = set()
    mux_copy = gate_list.copy()

    for i in range(num_controls):
        c.add(i + 1)

    if len(gate_list) > 1:
        nc, mux_copy = repetition_search(gate_list, num_controls)

    new_controls = {x for x in c if x not in nc}
    new_mux = [gate for gate in mux_copy if gate is not None]
    return new_controls, new_mux

def repetition_search(
        mux: list[NDArray[np.complex128]],
        level: int,
    ) -> tuple[set[int], list[NDArray[np.complex128]]]:
    """ Search for repetitions in the gate list.

    Parameters
    ----------
    `mux` : list[NDArray[np.complex128]]
        The list of gates.
    `level` : int
        The number of qubits.

    Returns
    -------
    `nc` : set[int]
        The set of removed controls.
    `mux_copy` : list[NDArray[np.complex128]]
        The new list of gates.
    """
    mux_copy = mux.copy()
    nc = set()
    d = 1

    while d <= len(mux) / 2:
        disentanglement = False
        if np.allclose(mux[d], mux[0]):
            mux_org = mux_copy.copy()
            repetitions = len(mux) / (2 * d)
            p = 0
            while repetitions > 0:
                repetitions -= 1
                valid, mux_copy = repetition_verify(p, d, mux, mux_copy)
                p = p + 2 * d
                if not valid:
                    mux_copy = mux_org
                    break
                if repetitions == 0:
                    disentanglement = True

        if disentanglement:
            removed_contr = level - np.log2(d)
            nc.add(removed_contr)
        d = 2 * d
    return nc, mux_copy

def repetition_verify(
        base,
        d,
        mux,
        mux_copy
    ) -> tuple[bool, list[NDArray[np.complex128]]]:
    """ Verify if the repetitions are valid.

    Parameters
    ----------
    `base` : int
        The base index.
    `d` : int
        The number of gates.
    `mux` : list[NDArray[np.complex128]]
        The list of gates.
    `mux_copy` : list[NDArray[np.complex128]]
        The new list of gates.

    Returns
    -------
    bool
        True if the repetitions are valid, False otherwise.
    `mux_copy` : list[NDArray[np.complex128]]
        The new list of gates.
    """
    i = 0
    next_base = base + d
    while i < d:
        if not np.allclose(mux[base], mux[next_base]):
            return False, mux_copy
        mux_copy[next_base] = None
        base, next_base, i = base + 1, next_base + 1, i + 1
    return True, mux_copy