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

""" Helper functions for quantum circuits.
"""

from __future__ import annotations

__all__ = [
    "decompose_multiplexor_rotations",
    "multiplexed_rz_angles",
    "extract_uvr_matrices",
    "extract_single_qubits_and_diagonal",
    "multiplexor_diagonal_matrix",
    "simplify",
    "repetition_search",
    "repetition_verify"
]

import numpy as np
from numpy.typing import NDArray

""" Constants for decomposing multiplexed RZ gates from Bergholm et al.
These are the (0, 0) and (1, 1) elements of the RZ gate matrix with angle -pi/2
Defined directly as complex numbers to avoid floating point errors
"""
SQRT2 = 1/np.sqrt(2)

RZ_PI2_00 = complex(
    SQRT2, SQRT2
)
RZ_PI2_11 = complex(
    SQRT2, -SQRT2
)


def decompose_multiplexor_rotations(
        angles: NDArray[np.float64],
        start_index: int,
        end_index: int,
        reverse_decomposition: bool
    ) -> NDArray[np.float64]:
    """ Recursively calculate the rotation angles for a multiplexed
    Pauli Rotation gate with a CX gate at the end of the circuit.

    Notes
    -----
    The rotation angles of the gates are stored in angles[start_index:end_index].
    If `reverse_decomposition` is `True`, the decomposition is performed such that
    a CX gate appears at the beginning of the circuit.

    Essentially, the circuit topology for the reversed decomposition is the reverse
    of the original decomposition.

    Parameters
    ----------
    `angles` : NDArray[np.float64]
        The list of rotation angles.
    `start_index` : int
        The start index of the rotation angles.
    `end_index` : int
        The end index of the rotation angles.
    `reverse_decomposition` : bool
        If True, decompose the gate such that there is a CX gate
        at the start of the circuit.

    Returns
    -------
    `angles` : NDArray[np.float64]
        The list of rotation angles.
    """
    mid_index = (end_index + start_index) // 2

    # Decompose the first half of the interval
    # and update the angles in place based on Shende's decomposition
    for i in range(start_index, mid_index):
        if reverse_decomposition:
            angles[i + mid_index - start_index], angles[i] = (
                (angles[i] + angles[i + mid_index - start_index]) / 2.0,
                (angles[i] - angles[i + mid_index - start_index]) / 2.0
            )
        else:
            angles[i], angles[i + mid_index - start_index] = (
                (angles[i] + angles[i + mid_index - start_index]) / 2.0,
                (angles[i] - angles[i + mid_index - start_index]) / 2.0
            )

    # Base case to stop the recursion
    if mid_index - start_index <= 1:
        return angles

    # Recursively decompose the second half of the interval
    # The second half is always decomposed in reverse order
    decompose_multiplexor_rotations(angles, start_index, mid_index, False)
    decompose_multiplexor_rotations(angles, mid_index, end_index, True)

    return angles

def multiplexed_rz_angles(
        phi_1: float,
        phi_2: float
    ) -> tuple[float, float]:
    """ Extract a RZ rotation (angle given by first output) such that

    .. math::

        e^{j \times phase}\times RZ(\theta) =
            \begin{pmatrix}
                e^{j\phi_1} & 0 \\
                0 & e^{j\phi_2}
            \end{pmatrix}

    The angles are used to create a multiplexed RZ gate, which is used
    to construct the diagonal gate based on Shende's decomposition [1].

    For more details, see the paper by Shende et al.
    [1] Shende, Bullock, Markov.
    Synthesis of Quantum Logic Circuits (2006)
    https://arxiv.org/abs/quant-ph/0406176

    Parameters
    ----------
    `phi_1` : float
        The first phase angle.
    `phi_2` : float
        The second phase angle.

    Returns
    -------
    tuple[float, float]
        The phase angle and the RZ angle.
    """
    phase = (phi_1 + phi_2) / 2.0
    rz_angle = phi_2 - phi_1
    return phase, rz_angle

def extract_uvr_matrices(
        a: NDArray[np.complex128],
        b: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """ Extract the matrices u, v, and r from unitary gates a, b for constructing
    a two qubit gate $F^1_2(U(2))$. This is used to construct the multiplexed gate
    $F^k_t(U(2))$.

    Notes
    -----
    This function is used to extract u, v, and r to construct a multiplexed gate
    based on the paper by Bergholm et al. For more details, see the paper [1],
    specifically section III.

    [1] Bergholm, Vartiainen, Möttönen, Salomaa,
    Quantum circuits with uniformly controlled one-qubit gates (2005).
    https://arxiv.org/pdf/quant-ph/0410066

    Parameters
    ----------
    `a` : NDArray[np.complex128]
        The first single qubit gate.
    `b` : NDArray[np.complex128]
        The second single qubit gate.

    Returns
    -------
    `v` : NDArray[np.complex128]
        The 2x2 unitary matrix v based on (eq.7).
    `u` : NDArray[np.complex128]
        The 2x2 unitary similarity transformation matrix u.
    `r` : NDArray[np.complex128]
        The diagonal matrix r.
    """
    # Hermitian conjugate of b (Eq 6)
    X = a @ np.conj(b).T

    # Determinant and phase of x
    det_X = np.linalg.det(X)
    X_11 = X[0, 0] / np.sqrt(det_X)
    phi = np.angle(det_X)

    # Compute the diagonal matrix r
    arg_X_11 = np.angle(X_11)

    # The implementation of the diagonal matrix r is
    # given below, but it can be chosen freely
    r_1 = np.exp(1j / 2 * ((np.pi - phi)/2 - arg_X_11))
    r_2 = np.exp(1j / 2 * ((np.pi - phi)/2 + arg_X_11 + np.pi))
    r = np.array([
        [r_1, 0],
        [0, r_2]
    ])

    # Eigendecomposition of r @ x @ r (Eq 8)
    # This is done via reforming Eq 6 to be similar to an eigenvalue decomposition
    rxr = r @ X @ r
    eigenvalues, u = np.linalg.eig(rxr)

    # Put the eigenvalues into a diagonal form
    diagonal = np.diag(np.sqrt(eigenvalues))

    # Handle specific case where the eigenvalue is near -i
    if np.abs(diagonal[0, 0] + 1j) < 1e-10:
        diagonal = np.flipud(diagonal)
        u = np.fliplr(u)

    # Calculate v based on the decomposition (Eq 7)
    v = diagonal @ np.conj(u).T @ np.conj(r).T @ b

    return v, u, r

def extract_single_qubits_and_diagonal(
        single_qubit_gates: list[NDArray[np.complex128]],
        num_qubits: int
    ) -> tuple[list[NDArray[np.complex128]], NDArray[np.complex128]]:
    """ Get the single qubit gates and diagonal arising in the decomposition
    of multiplexed gates given in the paper by Bergholm et al.

    Notes
    -----
    This function is used to decompose multiplexed gates based on
    the paper Eq. 14.

    [1] Bergholm, Vartiainen, Möttönen, Salomaa,
    Quantum circuits with uniformly controlled one-qubit gates (2005).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330

    Parameters
    ----------
    `single_qubit_gates` : list[NDArray[np.complex128]]
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
    # Copy the single qubit gates to avoid modifying the original gates
    single_qubit_gates = [np.copy(gate) for gate in single_qubit_gates]

    # Initialize the diagonal matrix
    diagonal = np.ones(2**num_qubits, dtype=np.complex128)

    # The number of target indices is 1
    num_controls = num_qubits - 1

    # Extract the single qubit gates and the diagonal gate for constructing
    # the multiplexed gate based on [1]
    for decomposition_step in range(num_controls):
        # For each decomposition step, there are 2^decomposition_step multiplexors
        # where each multiplexor has 2^(num_controls - decomposition_step) gates
        num_multiplexors = 2**decomposition_step

        for multiplexor_index in range(num_multiplexors):
            len_multiplexor = 2**(num_controls - decomposition_step)

            for i in range(len_multiplexor // 2):
                shift = multiplexor_index * len_multiplexor

                # Define a, b for F^1_2(U(2)) construction
                a = single_qubit_gates[shift + i]
                b = single_qubit_gates[shift + len_multiplexor // 2 + i]

                v, u, r = extract_uvr_matrices(a, b)

                # Replace the single qubit gates with v and u per figure 4
                single_qubit_gates[shift + i] = v
                single_qubit_gates[shift + len_multiplexor // 2 + i] = u

                # Decompose D gates per figure 3
                r_dagger = np.conj(r).T

                if multiplexor_index < num_multiplexors - 1:
                    k = shift + len_multiplexor + i
                    single_qubit_gates[k] = single_qubit_gates[k] @ r_dagger
                    single_qubit_gates[k] *= RZ_PI2_00
                    k += len_multiplexor // 2
                    single_qubit_gates[k] = single_qubit_gates[k] @ r
                    single_qubit_gates[k] *= RZ_PI2_11
                else:
                    for multiplexor_index_2 in range(num_multiplexors):
                        shift_2 = multiplexor_index_2 * len_multiplexor
                        k = 2 * (i + shift_2)
                        diagonal[k] *= r_dagger[0, 0] * RZ_PI2_00
                        diagonal[k + 1] *= r_dagger[1, 1] * RZ_PI2_00
                        k += len_multiplexor
                        diagonal[k] *= r[0, 0] * RZ_PI2_11
                        diagonal[k + 1] *= r[1, 1] * RZ_PI2_11

    return single_qubit_gates, diagonal

def multiplexor_diagonal_matrix(
        single_qubit_gates: list[NDArray[np.complex128]],
        num_qubits: int,
        simplified_controls: set[int]
    ) -> NDArray[np.complex128]:
    """ Get the diagonal matrix arising in the decomposition of multiplexor
    gates given in the paper by Bergholm et al.

    Notes
    -----
    This function to extract the diagonal matrix arising in the decomposition
    of multiplexed gates based on the paper by Bergholm et al.

    Parameters
    ----------
    `single_qubit_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.
    `num_qubits` : int
        The number of qubits.

    Returns
    -------
    NDArray[np.complex128]
        The diagonal matrix.
    """
    _, diagonal = extract_single_qubits_and_diagonal(single_qubit_gates, num_qubits)

    # Simplify the diagonal to minimize the number of controlled gates
    # needed to implement the diagonal gate
    if simplified_controls:
        control_qubits = sorted([num_qubits - i for i in simplified_controls], reverse=True)
        for i in range(num_qubits):
            if i not in [0] + control_qubits:
                step = 2**i
                diagonal = np.repeat(diagonal, 2, axis=0)
                for j in range(step, len(diagonal), 2 * step):
                    diagonal[j:j + step] = diagonal[j - step:j]

    return diagonal

def simplify(
        single_qubit_gates: list[NDArray[np.complex128]],
        num_controls: int
    ) -> tuple[set[int], list[NDArray[np.complex128]]]:
    """ Perform the multiplexor simplification.

    Notes
    -----
    The implementation of this simplification is based on the paper
    by by de Carvalho et al. [1]. The pseudocode is provided in Algorithm 1.

    [1] de Carvalho, Batista, de Veras, Araujo, da Silva,
    Quantum multiplexer simplification for state preparation (2024).
    https://arxiv.org/abs/2409.05618

    Parameters
    ----------
    `single_qubit_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.
    `num_controls` : int
        The number of controls.

    Returns
    -------
    `new_controls` : set[int]
        The new set of controls.
    `new_mux` : list[NDArray[np.complex128]]
        The new list of single qubit gates.
    """
    c: set[int] = set()
    nc: set[int] = set()
    mux_copy = single_qubit_gates.copy()

    # Add the position of the multiplexer controls to the set c
    for i in range(num_controls):
        c.add(i + 1)

    # Identify repetitions in the array and return the unnecessary
    # controls and a copy of the array, marking the repeated operators
    # as null
    if len(single_qubit_gates) > 1:
        nc, mux_copy = repetition_search(single_qubit_gates, num_controls)

    # Remove the unnecessary controls and the marked operators, creating
    # a new set of controls and a new array representing the simplified multiplexer
    controls_tree = {x for x in c if x not in nc}
    mux_tree = [gate for gate in mux_copy if gate is not None]

    return controls_tree, mux_tree

def repetition_search(
        multiplexor: list[NDArray[np.complex128]],
        level: int,
    ) -> tuple[set[int], list[NDArray[np.complex128]]]:
    """ Search for repetitions in the gate list.

    Notes
    -----
    The implementation of this simplification is based on the paper
    by by de Carvalho et al. [1]. The pseudocode is provided in Algorithm 2.

    [1] de Carvalho, Batista, de Veras, Araujo, da Silva,
    Quantum multiplexer simplification for state preparation (2024).
    https://arxiv.org/abs/2409.05618

    Parameters
    ----------
    `multiplexor` : list[NDArray[np.complex128]]
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
    mux_copy = multiplexor.copy()
    nc = set()
    d = 1

    # The positions of the multiplexer whose indices are a power of two
    # are traveled by the loop to find an operator identical to the
    # first one
    while d <= len(multiplexor) / 2:
        disentanglement = False

        # If a repeated angle is found, a copy of the array is created,
        # and we calculate in the next line the number of repetitions for
        # a possible pattern based on the distance between the two operators
        # and the size of the multiplexer
        if np.allclose(multiplexor[d], multiplexor[0]):
            mux_org = mux_copy.copy()
            repetitions = len(multiplexor) / (2 * d)
            p = 0

            # Each possible repetition is verified in the loop to confirm its
            # validity
            # We call the verification function for each of them, which returns
            # a boolean as true if the pattern is confirmed and marks the
            # copy of the array as described previously
            while repetitions > 0:
                repetitions -= 1
                valid, mux_copy = repetition_verify(p, d, multiplexor, mux_copy)
                p += 2 * d

                # If the repetitions are invalid, we restore the array to the original
                # and stop the search for this value of d
                if not valid:
                    mux_copy = mux_org
                    break

                # If all the repetitions for this pattern are confirmed, we have disentanglement,
                # and the position of the unnecessary control is calculated using the previous rule
                if repetitions == 0:
                    disentanglement = True

        # If disentanglement is true, we calculate the position of the control to be removed
        # and add it to the set of unnecessary controls
        if disentanglement:
            removed_control_index = level - np.log2(d)
            nc.add(removed_control_index)
        d *= 2

    return nc, mux_copy

def repetition_verify(
        base,
        d,
        multiplexor,
        mux_copy
    ) -> tuple[bool, list[NDArray[np.complex128]]]:
    """ Verify if the repetitions are valid. This is done by comparing each
    pair of operators with a distance d between them.

    If all of these pairs are identical, the repetition for this pattern is
    consistent. Operators marked for removal are flagged as null without
    changing the array size.

    Notes
    -----
    The implementation of this simplification is based on the paper
    by by de Carvalho et al. [1]. The pseudocode is provided in Algorithm 3.

    [1] de Carvalho, Batista, de Veras, Araujo, da Silva,
    Quantum multiplexer simplification for state preparation (2024).
    https://arxiv.org/abs/2409.05618

    Parameters
    ----------
    `base` : int
        The base index.
    `d` : int
        The number of gates.
    `multiplexor` : list[NDArray[np.complex128]]
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
        if not np.allclose(multiplexor[base], multiplexor[next_base]):
            return False, mux_copy
        mux_copy[next_base] = None
        base, next_base, i = base + 1, next_base + 1, i + 1

    return True, mux_copy