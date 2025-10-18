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

from __future__ import annotations

__all__ = [
    "generate_random_state",
    "generate_random_unitary",
    "generate_random_density_matrix",
    "generate_random_orthogonal_matrix",
    "generate_random_special_orthogonal_matrix",
    "generate_random_special_unitary_matrix"
]

import numpy as np
from numpy.typing import NDArray
from scipy.stats import unitary_group # type: ignore
from typing import Literal


def _generate_ginibre_matrix(
        num_rows: int,
        num_columns: int,
    ) -> NDArray[np.complex128]:
    """ Return a normally distributed complex random matrix.

    Parameters
    ----------
    `num_rows` : int
        Number of rows in output matrix.
    `num_columns` : int
        Number of columns in output matrix.

    Returns
    -------
    `ginibre_ensemble` : NDArray[np.complex128]
        A complex rectangular matrix where each real and imaginary
        entry is sampled from the normal distribution.
    """
    rng = np.random.default_rng()
    ginibre_ensemble = rng.normal(size=(num_rows, num_columns)) + 1j * rng.normal(
        size=(num_rows, num_columns)
    )
    return ginibre_ensemble

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

def generate_random_density_matrix(
        num_qubits: int,
        rank: int | None = None,
        generator: Literal["hilbert-schmidt", "bures"] = "hilbert-schmidt"
    ) -> NDArray[np.complex128]:
    """ Generate a random density matrix.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the density matrix.
    `rank` : int, optional, default=None
        The rank of the density matrix. If None, the matrix is full-rank,
        where rank is set to the number of qubits.
    `generator` : Literal["hilbert-schmidt", "bures"], optional, default="hilbert-schmidt"
        The method to use for generating the density matrix.
        Options are "hilbert-schmidt" or "bures".

    Returns
    -------
    NDArray[np.complex128]
        The generated random density matrix.

    Raises
    ------
    ValueError
        - If the `generator` is not recognized.
    """
    if not rank:
        rank = num_qubits

    if generator not in ["hilbert-schmidt", "bures"]:
        raise ValueError(f"Unrecognized generator method: {generator}")

    ginibre_ensemble = _generate_ginibre_matrix(2**num_qubits, 2**rank)

    if generator == "hilbert-schmidt":
        density_matrix = ginibre_ensemble @ ginibre_ensemble.conj().T
    elif generator == "bures":
        density_matrix = np.eye(2**num_qubits) + generate_random_unitary(num_qubits)
        density_matrix = density_matrix @ ginibre_ensemble
        density_matrix = density_matrix @ density_matrix.conj().T

    return density_matrix / np.trace(density_matrix)

def generate_random_orthogonal_matrix(num_qubits: int) -> NDArray[np.complex128]:
    """ Generate a random orthogonal matrix for the given number of qubits.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the orthogonal matrix.

    Returns
    -------
    `NDArray[np.complex128]`
        The random orthogonal matrix.
    """
    A = np.random.rand(2**num_qubits, 2**num_qubits)
    Q, R = np.linalg.qr(A)

    d = np.sign(np.diag(R))
    d[d == 0] = 1
    Q = Q @ np.diag(d)

    return Q.astype(np.complex128)

def generate_random_special_orthogonal_matrix(num_qubits: int) -> NDArray[np.float64]:
    """ Generate a random special orthogonal matrix for the given number of qubits.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the special orthogonal matrix.

    Returns
    -------
    `NDArray[np.float64]`
        The random special orthogonal matrix.
    """
    Q = generate_random_orthogonal_matrix(num_qubits)

    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    return Q.astype(np.float64)

def generate_random_special_unitary_matrix(num_qubits: int) -> NDArray[np.complex128]:
    """ Generate a random special unitary matrix for the given number of qubits.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the special unitary matrix.

    Returns
    -------
    `NDArray[np.complex128]`
        The random special unitary matrix.
    """
    U = generate_random_unitary(num_qubits)

    det = np.linalg.det(U)
    U = U / det**(1 / U.shape[0])

    return U