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

from __future__ import annotations

__all__ = [
    "cosine_similarity",
    "generate_random_state"
]

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance # type: ignore


def cosine_similarity(
        h1: dict[str, int],
        h2: dict[str, int]
    ) -> float:
    """ Calculate the cosine similarity between two histograms.

    Parameters
    ----------
    h1 : dict[str, int]
        The first histogram.
    h2 : dict[str, int]
        The second histogram.

    Returns
    -------
    float
        The cosine similarity between the two histograms.
    """
    # Convert dictionaries to lists
    keys = set(h1.keys()).union(h2.keys())
    dist_1 = [h1.get(key, 0) for key in keys]
    dist_2 = [h2.get(key, 0) for key in keys]

    return float(1 - distance.cosine(dist_1, dist_2))

def generate_random_state(num_qubits: int) -> NDArray[np.complex128]:
    """ Generate a random state vector.

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    Returns
    -------
    `statevector` : NDArray[np.complex128]
        The random state vector.
    """
    statevector = np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)
    statevector /= np.linalg.norm(statevector)
    return statevector