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
    "test_weyl_coordinates_simple",
    "test_weyl_coordinates_random"
]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
from scipy.stats import unitary_group

from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import weyl_coordinates

# Tolerance for floating point comparisons
INVARIANT_TOL = 1e-12

# Bell "Magic" basis
MAGIC = (
    1.0
    / np.sqrt(2)
    * np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex)
)


def two_qubit_local_invariants(U: NDArray[np.complex128]) -> NDArray[np.float64]:
    """ Calculate the local invariants for a two-qubit unitary.

    Notes
    -----
    This function calculates the local invariants for a two-qubit
    unitary as defined in Ref. 1. The local invariants are defined
    as [g1, g2, g3] where g1 = Tr(M^2) / (16 det(U_m)), g2 = (Tr(M^2) - Tr(M^4)) / (4 det(U_m)),
    and g3 = Tr(M^4) / (16 det(U_m)), where U_m = M^T M and M = U^T MAGIC U MAGIC^T.

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        Two-qubit unitary.

    Returns
    -------
    NDArray[np.float64]
        Local invariants [g1, g2, g3].
    """
    U = np.asarray(U)
    if U.shape != (4, 4):
        raise ValueError("Unitary must correspond to a two-qubit gate.")

    # Transform to bell basis
    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    # Get determinate since +- one is allowed.
    det_um = np.linalg.det(Um)
    M = Um.T.dot(Um)
    # trace(M)**2
    m_tr2 = M.trace()
    m_tr2 *= m_tr2

    # Table II of Ref. 1 or Eq. 28 of Ref. 2.
    G1 = m_tr2 / (16 * det_um)
    G2 = (m_tr2 - np.trace(M.dot(M))) / (4 * det_um)

    # Here we split the real and imag pieces of G1 into two so as
    # to better equate to the Weyl chamber coordinates (c0,c1,c2)
    # and explore the parameter space.
    # Also do a FP trick -0.0 + 0.0 = 0.0
    return np.round([G1.real, G1.imag, G2.real], 12) + 0.0

def local_equivalence(weyl: NDArray[np.complex128]) -> NDArray[np.float64]:
    """ Calculate the equivalent local invariants from the
    Weyl coordinates.

    Notes
    -----
    This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003),
    but we multiply weyl coordinates by 2 since we are
    working in the reduced chamber.

    Parameters
    ----------
    `weyl` : NDArray[np.float64]
        Weyl coordinates [c0, c1, c2].

    Returns
    -------
    NDArray[np.float64]
        Local invariants [g0, g1, g2].
    """
    g0_equiv = np.prod(np.cos(2 * weyl) ** 2) - np.prod(np.sin(2 * weyl) ** 2)
    g1_equiv = np.prod(np.sin(4 * weyl)) / 4
    g2_equiv = (
        4 * np.prod(np.cos(2 * weyl) ** 2)
        - 4 * np.prod(np.sin(2 * weyl) ** 2)
        - np.prod(np.cos(4 * weyl))
    )
    return np.round([g0_equiv, g1_equiv, g2_equiv], 12) + 0.0

def test_weyl_coordinates_simple() -> None:
    """ Check Weyl coordinates against known cases.
    """
    # Identity [0,0,0]
    U = np.identity(4).astype(complex)
    weyl = weyl_coordinates(U)
    assert_almost_equal(weyl, [0, 0, 0], decimal=8)

    # CNOT [pi/4, 0, 0]
    U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
    weyl = weyl_coordinates(U)
    assert_almost_equal(weyl, [np.pi / 4, 0, 0], decimal=8)

    # SWAP [pi/4, pi/4 ,pi/4]
    U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

    weyl = weyl_coordinates(U)
    assert_almost_equal(weyl, [np.pi / 4, np.pi / 4, np.pi / 4], decimal=8)

    # SQRT ISWAP [pi/8, pi/8, 0]
    U = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
            [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )

    weyl = weyl_coordinates(U)
    assert_almost_equal(weyl, [np.pi / 8, np.pi / 8, 0], decimal=8)

def test_weyl_coordinates_random() -> None:
    """ Randomly check Weyl coordinates with local invariants.
    """
    for _ in range(10):
        U = unitary_group.rvs(4).astype(complex)
        weyl = weyl_coordinates(U)
        local_equiv = local_equivalence(weyl.astype(float))
        local = two_qubit_local_invariants(U)
        assert_almost_equal(local, local_equiv)