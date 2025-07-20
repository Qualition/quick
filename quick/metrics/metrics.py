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
    "calculate_entanglement_entropy",
    "calculate_entanglement_entropy_slope",
    "calculate_hilbert_schmidt_test"
]

import numpy as np
from numpy.typing import NDArray
import quimb.tensor as qtn # type: ignore
from qiskit.quantum_info import partial_trace # type: ignore

from quick.predicates import is_density_matrix, is_statevector, is_unitary_matrix


def _calculate_1d_entanglement_range(mps: qtn.MatrixProductState) -> list[tuple[int, int]]:
    """ Get the entanglement range for entangled qubits in a 1D chain
    by checking the virtual (bond) dimensions of the tensors at each
    site in the MPS.

    Parameters
    ----------
    `mps` : qtn.MatrixProductState
        The MPS representation of the quantum state.

    Returns
    -------
    `entangled_blocks_indices` : list[tuple[int, int]]
        The indices of the MPS entangled blocks.
    """
    entangled_blocks_indices: list[tuple[int, int]] = []

    if mps.L == 1:
        return [(0, 0)]

    for site in range(mps.L):
        dim_left, dim_right = 1, 1

        # Define the dimensions for each site
        # The first and last sites are connected to only one site
        # as opposed to the other sites in the middle which are connected
        # to two sites to their left and right
        #  |
        #  ●━━ `dim_right`
        if site == 0:
            _, dim_right = mps[site].shape # type: ignore

        #              |
        # `dim_left` ━━●
        elif site == (mps.L - 1):
            dim_left, _ = mps[site].shape # type: ignore

        #              |
        # `dim_left` ━━●━━ `dim_right`
        else:
            dim_left, _, dim_right = mps[site].shape # type: ignore

        if dim_left < 2 and dim_right < 2:
            entangled_blocks_indices.append((site, site))
        elif dim_left < 2 and dim_right >= 2:
            temp = site
        elif dim_left >= 2 and dim_right < 2:
            entangled_blocks_indices.append((temp, site))

    return entangled_blocks_indices

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

    Raises
    ------
    ValueError
        - If the input is not a statevector.

    Usage
    -----
    >>> entanglements = get_entanglements(statevector)
    """
    if not is_statevector(statevector):
        raise ValueError("The input must be a statevector.")

    num_qubits = int(np.log2(statevector.size))

    # We need to have the statevector in MSB order for
    # correct extraction
    statevector = (
        statevector.reshape([2] * num_qubits)
        .transpose(list(range(num_qubits))[::-1])
        .flatten()
    )

    return _calculate_1d_entanglement_range(qtn.MatrixProductState.from_dense(statevector))

def calculate_shannon_entropy(probability_vector: NDArray[np.complex128]) -> float:
    """ Calculate the Shannon entropy of a probability vector.

    Parameters
    ----------
    `probability_vector` : NDArray[np.complex128]
        The probability vector.

    Returns
    -------
    float
        The Shannon entropy of the circuit.

    Usage
    -----
    >>> shannon_entropy = calculate_shannon_entropy(statevector)
    """
    probability_vector = probability_vector[(0 < probability_vector) & (probability_vector < 1)]
    return -np.sum(probability_vector * np.log2(probability_vector)).astype(float)

def calculate_entanglement_entropy(data: NDArray[np.complex128]) -> float:
    """ Calculate the Von Neumann entanglement entropy from the
    density matrix. In case of statevectors the entropy is simply
    0.

    Parameters
    ----------
    `data` : NDArray[np.complex128]
        The data, which can be a statevector or a density matrix.

    Returns
    -------
    float
        The entanglement entropy of the data.

    Raises
    ------
    ValueError
        - Input dimension matches a statevector but is not a valid statevector.
        - The input is not a valid density matrix.

    Usage
    -----
    >>> entanglement_entropy = calculate_entanglement_entropy(data)
    """
    # Handle the case of statevectors
    # and ensure the density matrix is
    # valid
    if data.ndim == 1:
        if is_statevector(data):
            return 0.0
        else:
            raise ValueError(
                "Input dimension matches a statevector "
                "but is not a valid statevector."
            )
    if data.ndim == 2:
        if data.shape[1] == 1:
            if is_statevector(data):
                return 0.0
            else:
                raise ValueError(
                    "Input dimension matches a statevector "
                    "but is not a valid statevector."
                )
        else:
            if not is_density_matrix(data):
                raise ValueError("The input is not a valid density matrix.")

    eigenvalues = np.maximum(np.real(np.linalg.eigvals(data)), 0.0)
    return calculate_shannon_entropy(eigenvalues)

def calculate_entanglement_entropy_slope(statevector: NDArray[np.complex128]) -> float:
    """ Calculate the slope of the entanglement entropy. This is
    used to determine whether a state is area-law or volume-law
    entangled, which is a measure of how the entanglement entropy
    scales with the number of qubits.

    If the slope is 1, which is a straight line, then the state is
    volume-law entangled. If the entropy decays after a while and
    forms a decaying curve, then the state is area-law entangled.

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The statevector of the circuit.

    Returns
    -------
    `slope` : float
        The slope of the entanglement entropy of the circuit.

    Raises
    ------
    ValueError
        - The input must be a statevector.

    Usage
    -----
    >>> entanglement_entropy_slope = calculate_entanglement_entropy_slope(statevector)
    """
    if not is_statevector(statevector):
        raise ValueError("The input must be a statevector.")

    num_qubits = int(
        np.ceil(
            np.log2(len(statevector))
        )
    )

    max_k = num_qubits // 2
    entropies = np.empty(max_k, dtype=np.float64)

    for k in range(1, max_k + 1):
        # Trace out rest of the qubits to extract the
        # reduced density matrix for the first k qubits
        rho_A = partial_trace(statevector, list(range(k, num_qubits))) # type: ignore
        S = calculate_entanglement_entropy(rho_A.data)
        entropies[k - 1] = S

    # We use half of the entropies to calculate the slope
    # for efficiency
    entropies = entropies[len(entropies) // 2:]
    x = np.arange(1, len(entropies) + 1)

    x_mean = np.mean(x)
    y_mean = np.mean(entropies)

    numerator = np.sum((x - x_mean) * (entropies - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator if denominator != 0 else 0

    return float(slope)

def calculate_hilbert_schmidt_test(
        unitary_1: NDArray[np.complex128],
        unitary_2: NDArray[np.complex128]
    ) -> float:
    """ Calculate the Hilbert-Schmidt test. This is a measure of the
    similarity of two unitary matrices.

    Parameters
    ----------
    `unitary_1` : NDArray[np.complex128]
        The first unitary matrix.

    `unitary_2` : NDArray[np.complex128]
        The second unitary matrix.

    Returns
    -------
    `chst` : float
        The Hilbert-Schmidt test of the two unitary matrices.

    Raises
    ------
    ValueError
        - If either of the matrices is not unitary.
        - If the matrices are not square.

    Usage
    -----
    >>> hilbert_schmidt_test = calculate_hilbert_schmidt_test(unitary_1, unitary_2)
    """
    if not is_unitary_matrix(unitary_1):
        raise ValueError("The first matrix is not unitary.")
    if not is_unitary_matrix(unitary_2):
        raise ValueError("The second matrix is not unitary.")

    num_qubits = int(np.log2(unitary_1.shape[0]))

    chst = 1/2**(2 * num_qubits) * np.abs(
        np.trace(
            np.dot(unitary_1.conj().T, unitary_2)
        )
    )**2

    return chst