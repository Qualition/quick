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

from __future__ import annotations

__all__ = [
    "generate_random_state",
    "generate_random_unitary"
]

import numpy as np
from numpy.typing import NDArray
from scipy.stats import unitary_group # type: ignore


def generate_random_state(num_qubits: int) -> NDArray[np.complex128]:
    """ Generate a random state vector for the given number of qubits.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the state vector.

    Returns
    -------
    `NDArray[np.complex128]`
        The random state vector.
    """
    unnormalized_state = np.random.rand(2 ** num_qubits) + 1j * np.random.rand(2 ** num_qubits)
    return unnormalized_state / np.linalg.norm(unnormalized_state)

def generate_random_unitary(num_qubits: int) -> NDArray[np.complex128]:
    """ Generate a random unitary matrix for the given number of qubits.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the unitary matrix.

    Returns
    -------
    `NDArray[np.complex128]`
        The random unitary matrix.
    """
    return unitary_group.rvs(2 ** num_qubits).astype(np.complex128)