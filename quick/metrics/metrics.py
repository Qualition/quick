# Copyright 2023-2025 Qualition Computing LLC.
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

""" Module for analysing circuits.
"""

from __future__ import annotations

__all__ = [
    "calculate_entanglement_range",
    "calculate_shannon_entropy",
    "calculate_entanglement_entropy"
]

import numpy as np
from numpy.typing import NDArray
import quimb.tensor as qtn # type: ignore


def _get_submps_indices(mps: qtn.MatrixProductState) -> list[tuple[int, int]]:
    """ Get the indices of contiguous blocks in the MPS. For testing purposes,
    this method is static.

    Notes
    -----
    Certain sites may not be entangled with the rest, and thus we can simply apply
    a single qubit gate to them as opposed to a two qubit gate.

    This reduces the overall cost of the circuit for a given layer. If all sites are
    entangled, then the method will simply return the indices of the MPS, i.e., for
    10 qubit system [(0, 9)]. If sites 0 and 1 are not entangled at all with the rest,
    the method will return [(0, 0), (1,1), (2, 9)].

    The implementation is based on the analytical decomposition [1].

    For more information, refer to the publication below:
    [1] Shi-Ju.
    Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates (2020).
    https://arxiv.org/abs/1908.07958

    Returns
    -------
    `submps_indices` : list[tuple[int, int]]
        The indices of the MPS contiguous blocks.

    Usage
    -----
    >>> mps.get_submps_indices()
    """
    sub_mps_indices: list[tuple[int, int]] = []

    if mps.L == 1:
        return [(0, 0)]

    for site in range(mps.L):
        # Reset the dimension variables for each iteration
        dim_left, dim_right = 1, 1

        # Define the dimensions for each site
        # The first and last sites are connected to only one site
        # as opposed to the other sites in the middle which are connected
        # to two sites to their left and right
        #
        #  |
        #  ●━━ `dim_right`
        if site == 0:
            _, dim_right = mps[site].shape # type: ignore
        #
        #              |
        # `dim_left` ━━●
        elif site == (mps.L - 1):
            dim_left, _ = mps[site].shape # type: ignore
        #
        #              |
        # `dim_left` ━━●━━ `dim_right`
        else:
            dim_left, _, dim_right = mps[site].shape # type: ignore

        if dim_left < 2 and dim_right < 2:
            sub_mps_indices.append((site, site))
        elif dim_left < 2 and dim_right >= 2:
            temp = site
        elif dim_left >= 2 and dim_right < 2:
            sub_mps_indices.append((temp, site))

    return sub_mps_indices

def calculate_entanglement_range(statevector: NDArray[np.complex128]) -> list[tuple[int, int]]:
    """ Get the entanglements of the circuit.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The statevector of the circuit.

    Returns
    -------
    list[tuple[int, int]]
        The entanglements of the circuit.

    Usage
    -----
    >>> entanglements = get_entanglements(statevector)
    """
    statevector = statevector.flatten()
    num_qubits = int(np.log2(statevector.size))

    # We need to have the statevector in MSB order for
    # correct extraction
    statevector = (
        statevector.reshape([2] * num_qubits)
        .transpose(list(range(num_qubits))[::-1])
        .flatten()
    )

    return _get_submps_indices(qtn.MatrixProductState.from_dense(statevector))

def calculate_shannon_entropy(statevector: NDArray[np.complex128]) -> float:
    """ Calculate the Shannon entropy.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The statevector of the circuit.

    Returns
    -------
    float
        The Shannon entropy of the circuit.

    Usage
    -----
    >>> shannon_entropy = calculate_shannon_entropy(statevector)
    """
    statevector = statevector[(0 < statevector) & (statevector < 1)]
    return -np.sum(statevector * np.log2(statevector)).astype(float)

def calculate_entanglement_entropy(statevector: NDArray[np.complex128]) -> float:
    """ Calculate the entanglement entropy of the circuit.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The statevector of the circuit.

    Returns
    -------
    float
        The entanglement entropy of the circuit.

    Usage
    -----
    >>> entanglement_entropy = calculate_entanglement_entropy(statevector)
    """
    density_matrix = np.outer(statevector, statevector.conj())
    eigenvalues = np.maximum(np.real(np.linalg.eigvals(density_matrix)), 0.0)
    return calculate_shannon_entropy(eigenvalues)