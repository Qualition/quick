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

""" Module for analysing circuits.
"""

from __future__ import annotations

__all__ = ["get_entanglements"]

import quimb.tensor as qtn # type: ignore

from quick.circuit import Circuit


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
    [1] Ran, Shi-Ju.
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

def get_entanglements(circuit: Circuit) -> list[tuple[int, int]]:
    """ Get the entanglements of the circuit.

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to extract the entanglements from.

    Returns
    -------
    list[tuple[int, int]]
        The entanglements of the circuit.

    Usage
    -----
    >>> circuit.get_entanglements()
    """
    # Copy the circuit to avoid modifying the original circuit
    qc = circuit.copy()
    qc.vertical_reverse()

    # Extract the statevector of the circuit
    # to determine the entanglements
    statevector = qc.get_statevector()

    # To find the entanglements, we need to convert the statevector
    # to a matrix product state (MPS) representation and then find
    # the sub-MPS indices
    return _get_submps_indices(qtn.MatrixProductState.from_dense(statevector))