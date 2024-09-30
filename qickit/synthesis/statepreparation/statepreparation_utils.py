# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the Qualition License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/QIMP/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Helper functions for the state preparation methods.
"""

from __future__ import annotations

__all__ = [
    "gray_code",
    "compute_alpha_y",
    "compute_alpha_z",
    "compute_m",
    "compute_control_indices",
    "bloch_angles",
    "rotations_to_disentangle",
    "k_s",
    "a",
    "b",
    "reverse_qubit_state",
    "find_squs_for_disentangling",
    "apply_ucg",
    "apply_diagonal_gate",
    "apply_diagonal_gate_to_diag",
    "apply_multi_controlled_gate",
    "ucg_is_identity_up_to_global_phase",
    "merge_ucgate_and_diag",
    "construct_basis_states",
    "diag_is_identity_up_to_global_phase",
    "get_binary_rep_as_list",
    "get_qubits_by_label"
]

from collections.abc import Sequence
from itertools import product
import numpy as np
from numpy.typing import NDArray


# Episilon value for floating point comparisons
EPSILON = 1e-10


""" Helper functions for the Mottonen encoder
"""

def gray_code(index: int) -> int:
    """ Return Gray code at the specified index.

    Parameters
    ----------
    `index`: int
        The index of the Gray code to return.

    Returns
    -------
    int
        The Gray code at the specified index.
    """
    return index ^ (index >> 1)

def compute_alpha_y(
        magnitude: NDArray[np.float64],
        k: int,
        j: int
    ) -> float:
    """ Return the rotation angle required for encoding the real components of the state
    at the specified indices.

    Notes
    -----
    This is the implementation of Equation (8) in the reference.
    Note the off-by-1 issues (the paper is 1-based).

    Parameters
    ----------
    `magnitude` : NDArray[np.float64]
        The magnitude of the state.
    `k` : int
        The index of the current qubit.
    `j` : int
        The index of the current angle.

    Returns
    -------
    float
        The rotation angles required for encoding the real components of the state
        at the specified indices.
    """
    m = 2 ** (k - 1)
    enumerator = sum(
        magnitude[(2 * (j + 1) - 1) * m + bit] ** 2 \
            for bit in range(m)
    )

    m = 2**k
    divisor = sum(
        magnitude[j * m + bit] ** 2 \
            for bit in range(m)
    )

    if divisor != 0:
        return 2 * np.arcsin(np.sqrt(enumerator / divisor)) # type: ignore
    return 0

def compute_alpha_z(
        phase: NDArray[np.float64],
        k: int,
        j: int
    ) -> float:
    """ Compute the angles alpha_k for the z rotations.

    Notes
    -----
    This is the implementation of Equation (5) in the reference.
    Note the off-by-1 issues (the paper is 1-based).

    Parameters
    ----------
    `phase` : NDArray[np.float64]
        The phase of the state.
    `k` : int
        The index of the current qubit.
    `j` : int
        The index of the current angle.

    Returns
    -------
    float
        The rotation angles required for encoding the imaginary components of the state
        at the specified indices.
    """
    m = 2 ** (k - 1)
    base1 = (2 * (j + 1) - 1) * m
    base2 = (2 * (j + 1) - 2) * m
    diff_sum = 0.0
    for bit in range(m):
        diff_sum += phase[base1 + bit] - phase[base2 + bit]
    return diff_sum / m

def compute_m(k: int) -> NDArray[np.float64]:
    """ Compute matrix M which takes alpha -> theta.

    Notes
    -----
    This is the implementation of Equation (3) in the reference.

    Parameters
    ----------
    `k` : int
        The number of qubits.

    Returns
    -------
    `m` : NDArray[np.float64]
        The matrix M which takes alpha -> theta.
    """
    n = 2**k
    m = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            m[i, j] = (-1) ** bin(j & gray_code(i)).count("1") * 2 ** (-k)
    return m

def compute_control_indices(index: int) -> list[int]:
    """ Return the control indices for the CX gates.

    Notes
    -----
    This code implements the control qubit indices following
    Fig 2 in the reference in a recursive manner. The secret
    to success is to 'kill' the last token in the recursive call.

    Parameters
    ----------
    `index` : int
        The index of the control qubit.

    Returns
    -------
    list[int]
        The control indices for the CX gates.
    """
    if index == 0:
        return []
    side = compute_control_indices(index - 1)[:-1]
    return side + [index - 1] + side + [index - 1]


""" Helper functions for the Shende encoder
"""

def bloch_angles(pair_of_complex: Sequence[complex]) -> tuple:
    """ Take a pair of complex numbers and return the corresponding Bloch angles.

    Parameters
    ----------
    `pair_of_complex` : Sequence[complex]
        The list of complex numbers.

    Returns
    -------
    tuple
        The list of Bloch angles.
    """
    [a_complex, b_complex] = pair_of_complex
    a_complex = complex(a_complex)
    b_complex = complex(b_complex)

    a_magnitude = abs(a_complex)
    b_magnitude = abs(b_complex)

    final_r = np.sqrt(a_magnitude ** 2 + b_magnitude ** 2)

    if final_r < 1e-10:
        theta, phi, final_r, final_t = 0.0, 0.0, 0.0, 0.0
    else:
        theta = 2 * np.arccos(a_magnitude / final_r)
        a_arg = float(np.angle(a_complex))
        b_arg = float(np.angle(b_complex))
        final_t = a_arg + b_arg
        phi = b_arg - a_arg

    return final_r * np.exp(1.0j * final_t / 2), theta, phi

def rotations_to_disentangle(local_param: NDArray[np.complex128]) -> tuple:
    """ Return Ry and Rz rotation angles used to disentangle the LSB qubit.
    These rotations make up the block diagonal matrix U (i.e. multiplexor)
    that disentangles the LSB.

    Parameters
    ----------
    `local_param` : NDArray[np.complex128]
        The list of local parameters.

    Returns
    -------
    tuple
        The tuple of global parameters.
    """
    remaining_vector = []
    thetas = []
    phis = []
    param_len = len(local_param)

    for i in range(param_len // 2):
        # Apply Ry and Rz rotations to transition the Bloch vector from 0 to the "imaginary" qubit state
        # This is conceptualized as a qubit state defined by amplitudes at indices 2*i and 2*(i+1),
        # which correspond to the selected qubits of the multiplexor being in state |i>
        (remains, add_theta, add_phi) = bloch_angles(local_param[2 * i : 2 * (i + 1)]) # type: ignore
        remaining_vector.append(remains)

        # Perform rotations on all imaginary qubits of the full vector to transition
        # their state towards zero, indicated by the negative sign
        thetas.append(-add_theta)
        phis.append(-add_phi)

    return remaining_vector, thetas, phis


""" Helper functions for the Isometry encoder
"""

def k_s(k: int, s: int) -> int:
    return (k >> s) & 1

def a(k: int, s: int) -> int:
    return k // 2**s

def b(k: int, s: int) -> int:
    return k - a(k, s) * 2**s

def reverse_qubit_state(
        state: NDArray[np.complex128],
        basis_state: int
    ) -> NDArray[np.complex128]:
    """ Return the reverse of the qubit state.

    Parameters
    ----------
    `state` : NDArray[np.complex128]
        The qubit state.
    `basis_state` : int
        The basis state.

    Returns
    -------
    NDArray[np.complex128]
        The reverse of the qubit state
    """
    norm = np.linalg.norm(state)
    if norm < EPSILON:
        return np.eye(2).astype(complex)
    norm_inv = 1.0 / norm
    if basis_state == 0:
        return np.array([
            [np.conj(state[0]) * norm_inv, np.conj(state[1]) * norm_inv],
            [-state[1] * norm_inv, state[0] * norm_inv]
        ])
    else:
        return np.array([
            [-state[1] * norm_inv, state[0] * norm_inv],
            [np.conj(state[0]) * norm_inv, np.conj(state[1]) * norm_inv]
        ])

def find_squs_for_disentangling(
        v: NDArray[np.complex128],
        k: int,
        s: int,
        n: int
    ) -> list[NDArray[np.complex128]]:
    """ Return the list of single qubit unitaries for disentangling qubits.

    Parameters
    ----------
    `v` : NDArray[np.complex128]
        The qubit state.
    `k` : int
        The index of the qubit.
    `s` : int
        The index of the qubit.
    `n` : int
        The number of qubits.

    Returns
    -------
    list[NDArray[np.complex128]]
        The list of single qubit unitaries for disentangling qubits.
    """
    if b(k, s+1) == 0:
        i_start = a(k, s+1)
    else:
        i_start = a(k, s+1) + 1
    output = [np.eye(2).astype(complex) for _ in range(i_start)]
    single_qubit_unitaries = []
    for i in range(i_start, 2**(n-s-1)):
        state = np.array([
            v[2 * i * 2**s + b(k, s), 0],
            v[(2 * i + 1) * 2**s + b(k, s), 0]
        ])
        single_qubit_unitaries.append(reverse_qubit_state(state, k_s(k, s)))
    output.extend(single_qubit_unitaries)
    return output

def apply_ucg(
        m: NDArray[np.complex128],
        k: int,
        single_qubit_gates: list[NDArray[np.complex128]]
    ) -> NDArray[np.complex128]:
    """ Apply the unitary controlled gate to the matrix.

    Parameters
    ----------
    `m` : NDArray[np.complex128]
        The matrix.
    `k` : int
        The index of the qubit.
    `single_qubit_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.

    Returns
    -------
    NDArray[np.complex128]
        The matrix after applying the unitary controlled gate.
    """
    m = m.copy()
    num_qubits = int(np.log2(m.shape[0]))
    spacing = 2**(num_qubits - k - 1)
    for j in range(2**(num_qubits - 1)):
        i = (j // spacing) * spacing + j
        gate_index = i // 2**(num_qubits - k)
        for col in range(m.shape[1]):
            a, b = m[i, col], m[i + spacing, col]
            gate = single_qubit_gates[gate_index]
            m[i, col] = gate[0, 0] * a + gate[0, 1] * b
            m[i + spacing, col] = gate[1, 0] * a + gate[1, 1] * b
    return m

def apply_diagonal_gate(
        m: NDArray[np.complex128],
        action_qubit_labels: list[int],
        diagonal: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
    """ Apply the diagonal gate to the matrix.

    Parameters
    ----------
    `m` : NDArray[np.complex128]
        The matrix.
    `action_qubit_labels` : list[int]
        The list of action qubit labels.
    `diagonal` : NDArray[np.complex128]
        The diagonal matrix.

    Returns
    -------
    NDArray[np.complex128]
        The matrix after applying the diagonal gate.
    """
    m = m.copy()
    num_qubits = int(np.log2(m.shape[0]))
    for state in product([0, 1], repeat=num_qubits):
        diagonal_index = sum(state[i] << (len(action_qubit_labels) - 1 - idx) for idx, i in enumerate(action_qubit_labels))
        i = sum(state[j] << (num_qubits - 1 - j) for j in range(num_qubits))
        m[i, :] *= diagonal[diagonal_index]
    return m

def apply_diagonal_gate_to_diag(
        m_diagonal: NDArray[np.complex128],
        action_qubit_labels: list[int],
        diagonal: NDArray[np.complex128],
        num_qubits: int
    ) -> NDArray[np.complex128]:
    """ Apply the diagonal gate to the diagonal matrix.

    Parameters
    ----------
    `m_diagonal` : NDArray[np.complex128]
        The diagonal matrix.
    `action_qubit_labels` : list[int]
        The list of action qubit labels.
    `diagonal` : NDArray[np.complex128]
        The diagonal matrix.
    `num_qubits` : int
        The number of qubits.

    Returns
    -------
    NDArray[np.complex128]
        The diagonal matrix after applying the diagonal gate.
    """
    if not m_diagonal:
        return m_diagonal
    for state in product([0, 1], repeat=num_qubits):
        diagonal_index = sum(state[i] << (len(action_qubit_labels) - 1 - idx) for idx, i in enumerate(action_qubit_labels))
        i = sum(state[j] << (num_qubits - 1 - j) for j in range(num_qubits))
        m_diagonal[i] *= diagonal[diagonal_index]
    return m_diagonal

def apply_multi_controlled_gate(
        m: NDArray[np.complex128],
        control_labels: list[int],
        target_label: int,
        gate: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
    """ Apply the multi-controlled gate to the matrix.

    Parameters
    ----------
    `m` : NDArray[np.complex128]
        The matrix.
    `control_labels` : list[int]
        The list of control labels.
    `target_label` : int
        The target label.
    `gate` : NDArray[np.complex128]
        The gate.

    Returns
    -------
    NDArray[np.complex128]
        The matrix after applying the multi-controlled gate.
    """
    m = m.copy()
    num_qubits = int(np.log2(m.shape[0]))
    control_set = set(control_labels)
    free_qubits = num_qubits - len(control_labels) - 1
    for state_free in product([0, 1], repeat=free_qubits):
        e1, e2 = construct_basis_states(state_free, control_set, target_label)
        for i in range(m.shape[1]):
            temp = gate @ np.array([[m[e1, i]], [m[e2, i]]])
            m[e1, i], m[e2, i] = temp[0, 0], temp[1, 0]
    return m

def ucg_is_identity_up_to_global_phase(single_qubit_gates: list[NDArray[np.complex128]]) -> bool:
    """ Check if the unitary controlled gate is identity up to global phase.

    Parameters
    ----------
    `single_qubit_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.

    Returns
    -------
    bool
        True if the unitary controlled gate is identity up to global phase, False otherwise.
    """
    if np.abs(single_qubit_gates[0][0, 0]) < EPSILON:
        return False
    global_phase = 1 / single_qubit_gates[0][0, 0]
    for gate in single_qubit_gates:
        if not np.allclose(gate * global_phase, np.eye(2).astype(complex), atol=EPSILON):
            return False
    return True

def merge_ucgate_and_diag(
        single_qubit_gates: list[NDArray[np.complex128]],
        diagonal: NDArray[np.complex128]
    ) -> list[NDArray[np.complex128]]:
    """ Merge the unitary controlled gate and the diagonal matrix.

    Parameters
    ----------
    `single_qubit_gates` : list[NDArray[np.complex128]]
        The list of single qubit gates.
    `diagonal` : NDArray[np.complex128]
        The diagonal matrix.

    Returns
    -------
    list[NDArray[np.complex128]]
        The list of single qubit gates after merging the unitary controlled gate and the diagonal matrix.
    """
    return [
        np.dot(np.array([[diagonal[2 * i], complex(0, 0)], [complex(0, 0), diagonal[2 * i + 1]]]), gate)
        for i, gate in enumerate(single_qubit_gates)
    ]

def construct_basis_states(
        state_free: tuple[int, ...],
        control_set: set[int],
        target_label: int
    ) -> tuple[int, int]:
    """ Construct the basis states.

    Parameters
    ----------
    `state_free` : tuple[int, ...]
        The tuple of free states.
    `control_set` : set[int]
        The set of control states.
    `target_label` : int
        The target label.

    Returns
    -------
    tuple[int, int]
        The basis states.
    """
    size = len(state_free) + len(control_set) + 1
    e1 = e2 = 0
    j = 0
    for i in range(size):
        e1 <<= 1
        e2 <<= 1
        if i in control_set:
            e1 += 1
            e2 += 1
        elif i == target_label:
            e2 += 1
        else:
            e1 += state_free[j]
            e2 += state_free[j]
            j += 1
    return e1, e2

def diag_is_identity_up_to_global_phase(diagonal: NDArray[np.complex128]) -> bool:
    """ Check if the diagonal matrix is identity up to global phase.

    Parameters
    ----------
    `diagonal` : NDArray[np.complex128]
        The diagonal matrix.

    Returns
    -------
    bool
        True if the diagonal matrix is identity up to global phase, False otherwise.
    """
    if np.abs(diagonal[0]) >= EPSILON:
        global_phase = 1.0 / diagonal[0]
    else:
        return False

    for d in diagonal:
        if np.abs(global_phase * d - 1.0) >= EPSILON:
            return False

    return True

def get_binary_rep_as_list(
        n: int,
        num_digits: int
    ) -> list[int]:
    """ Return the binary representation of the number as a list.

    Parameters
    ----------
    `n` : int
        The number.
    `num_digits` : int
        The number of digits.

    Returns
    -------
    list[int]
        The binary representation of the number as a list.
    """
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary[-num_digits:]

def get_qubits_by_label(
        labels: list[int],
        qubits: list[int],
        num_qubits: int
    ) -> list[int]:
    """ Return the qubits by label.

    Parameters
    ----------
    `labels` : list[int]
        The list of labels.
    `qubits` : list[int]
        The list of qubits.
    `num_qubits` : int
        The number of qubits.

    Returns
    -------
    list[int]
        The qubits by label.
    """
    return [qubits[num_qubits - label - 1] for label in labels]