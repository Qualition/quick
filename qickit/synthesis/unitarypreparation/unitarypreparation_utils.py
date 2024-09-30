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

""" Helper functions for the unitary preparation methods.
"""

from __future__ import annotations

__all__ = ["phase_matrix",
           "rotation_matrix",
           "deconstruct_single_qubit_matrix_into_angles"]

import cmath
import math
import numpy as np
from numpy.typing import NDArray


""" Helper functions for Shannon Decomposition encoder
"""

def phase_matrix(angle: float) -> NDArray[np.complex128]:
    """ Return a phase matrix.

    Parameters
    ----------
    `angle` : float
        The phase angle.

    Returns
    -------
    NDArray[np.complex128]
        The phase matrix.
    """
    return np.diag([1, np.exp(1j * angle)])

def rotation_matrix(angle: float) -> NDArray[np.complex128]:
    """ Return a rotation matrix.

    Parameters
    ----------
    `angle` : float
        The rotation angle.

    Returns
    -------
    NDArray[np.complex128]
        The rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

def deconstruct_single_qubit_matrix_into_angles(matrix: NDArray[np.complex128]) -> tuple[float, float, float]:
    """ Break down a 2x2 unitary into ZYZ angle parameters.

    Notes
    -----
    Given a unitary U, this function returns three angles: $\phi_0, \phi_1, \phi_2$,
    such that:  $U = Z^{\phi_2 / \pi} Y^{\phi_1 / \pi} Z^{\phi_0/ \pi}$
    for the Pauli matrices Y and Z. That is, phasing around Z by $\phi_0$ radians,
    then rotating around Y by $\phi_1$ radians, and then phasing again by
    $\phi_2$ radians will produce the same effect as the original unitary.
    (Note that the matrices are applied right to left.)

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        The 2x2 unitary matrix to break down.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the amount to phase around Z, then rotate around Y,
        then phase around Z (all in radians).
    """
    # Anti-cancel left-vs-right phase along top row
    right_phase = cmath.phase(matrix[0, 1] * np.conj(matrix[0, 0])) + math.pi
    matrix = np.dot(matrix, phase_matrix(-right_phase))

    # Cancel top-vs-bottom phase along left column
    bottom_phase = cmath.phase(matrix[1, 0] * np.conj(matrix[0, 0]))
    matrix = np.dot(phase_matrix(-bottom_phase), matrix)

    # Lined up for a rotation, clear the off-diagonal cells with one
    rotation = math.atan2(abs(matrix[1, 0]), abs(matrix[0, 0]))
    matrix = np.dot(rotation_matrix(-rotation), matrix)

    # Cancel top-left-vs-bottom-right phase
    diagonal_phase = cmath.phase(matrix[1, 1] * np.conj(matrix[0, 0]))

    # Ignoring global phase, return the three angles
    return right_phase + diagonal_phase, rotation * 2, bottom_phase