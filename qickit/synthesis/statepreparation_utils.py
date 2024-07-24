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

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray


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

def compute_alpha_y(magnitude: Sequence[float],
                    k: int,
                    j: int) -> float:
    """ Return the rotation angle required for encoding the real components of the state
    at the specified indices.

    Notes
    -----
    This is the implementation of Equation (8) in the reference.
    Note the off-by-1 issues (the paper is 1-based).

    Parameters
    ----------
    `magnitude` : Sequence[float]
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
        return 2 * np.arcsin(np.sqrt(enumerator / divisor))
    return 0

def compute_alpha_z(phase: NDArray[np.float64],
                    k: int,
                    j: int) -> float:
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
    ind1 = [(2 * (j + 1) - 1) * m + bit for bit in range(m)]
    ind2 = [(2 * (j + 1) - 2) * m + bit for bit in range(m)]
    diff = (phase[ind1] - phase[ind2]) / m
    return sum(diff)

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
    `local_param` : NDArray[float]
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