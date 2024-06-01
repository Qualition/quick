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

import numpy as np
from numpy.typing import NDArray

# Import `qickit.types.Collection`
from qickit.types import Collection


""" Helper functions for the Mottonen encoder
"""

def grayCode(x: int) -> int:
    """ Return the gray code of x.

    Parameters
    ----------
    `x` : int
        The number.

    Returns
    -------
    int
        The gray code of x.
    """
    return x ^ (x >> 1)

def alpha_y(angles: Collection[float],
            num_qubits: int,
            angle_index: int,) -> float:
    """ Return the alpha required for the amplitude encoding circuit.

    Parameters
    ----------
    `angles` : Collection[float]
        The array of angles.
    `num_qubits` : int
        The number of qubits.
    `angle_index` : int
        The index of the angle.

    Returns
    -------
    float
        The alpha required for the amplitude encoding circuit.
    """
    abs_angles = np.abs(list(angles))
    m = int(2**(num_qubits - 1))
    a = 0

    for i in range(m):
        a += abs_angles[(2 * (angle_index + 1) - 1) * m + i]**2

    mk = int(2**num_qubits)
    b = 0

    for i in range(mk):
        b += abs_angles[angle_index * mk + i]**2

    if b != 0 :
        ratio = np.sqrt(a / b)
    else :
        ratio = 0.0

    return 2*np.arcsin(ratio)

def M(k: int) -> NDArray[np.float64]:
    """ Return the matrix M required for the amplitude encoding circuit.

    Parameters
    ----------
    `k` : int
        The number of qubits.

    Returns
    -------
    NDArray[np.float64]
        The matrix M required for the amplitude encoding circuit.
    """
    n = 2**k
    M = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            M[i, j] = 2**(-k) * (-1)**(bin(j & grayCode(i)).count("1"))

    return M

def theta(M: NDArray[np.float64],
          alphas: list[float],) -> NDArray[np.float64]:
    """ Return the theta required for the amplitude encoding circuit.

    Parameters
    ----------
    `M` : NDArray[np.float_]
        The matrix M required for the amplitude encoding circuit.
    `alphas` : list[float]
        The array of alphas.

    Returns
    -------
    NDArray[np.float64]
        The theta required for the amplitude encoding circuit.
    """
    return M @ alphas

def ind(k: int) -> list[int]:
    """ Return the index required for the amplitude encoding circuit.

    Parameters
    ----------
    `k` : int
        The number of qubits.

    Returns
    -------
    list[int]
        The indices required for the amplitude encoding circuit.
    """
    n = 2**k
    code = [grayCode(i) for i in range(n)]

    control = []

    for i in range(n - 1):
        control.append(int(np.log2(code[i]^code[i + 1])))
    control.append(int(np.log2(code[n - 1]^code[0])))
    return control

""" Helper functions for the Shende encoder
"""

def bloch_angles(pair_of_complex: Collection[complex]) -> tuple:
    """ Take a pair of complex numbers and return the corresponding Bloch angles.

    Parameters
    ----------
    `pair_of_complex` : Collection[complex]
        The list of complex numbers.

    Returns
    -------
    tuple
        The list of Bloch angles.
    """
    [a_complex, b_complex] = pair_of_complex
    a_complex = complex(a_complex)
    b_complex = complex(b_complex)
    mag_a = abs(a_complex)
    final_r = np.sqrt(mag_a**2 + np.absolute(b_complex) ** 2)

    if final_r < 1e-10:
        theta, phi, final_r, final_t = 0.0, 0.0, 0.0, 0.0

    else:
        theta = 2 * np.arccos(mag_a / final_r)
        a_arg = float(np.angle(a_complex))
        b_arg = float(np.angle(b_complex))
        final_t = a_arg + b_arg
        phi = b_arg - a_arg

    return final_r * np.exp(1.0j * final_t / 2), theta, phi

def rotations_to_disentangle(local_param: Collection[complex]) -> tuple:
    """ Take a list of local parameters and return a list of global parameters.

    Parameters
    ----------
    `local_param` : Collection[float]
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
        (remains, add_theta, add_phi) = bloch_angles(local_param[2 * i : 2 * (i + 1)])
        remaining_vector.append(remains)
        thetas.append(-add_theta)
        phis.append(-add_phi)

    return remaining_vector, thetas, phis