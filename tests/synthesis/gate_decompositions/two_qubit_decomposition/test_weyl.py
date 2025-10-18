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

__all__ = ["TestWeyl"]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
from scipy.stats import unitary_group

from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import (
    M,
    M_DAGGER,
    weyl_coordinates
)

# Tolerance for floating point comparisons
INVARIANT_TOL = 1e-12

# Constants
PI = np.pi
PI_DOUBLE = 2 * PI
PI2 = PI / 2
PI4 = PI / 4


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
    U_magic_basis = M_DAGGER @ U @ M

    # Get det since +- one is allowed.
    det_um = np.linalg.det(U_magic_basis)
    M_squared = U_magic_basis.T @ U_magic_basis

    # trace(M)**2
    m_tr2 = M_squared.trace()
    m_tr2 *= m_tr2

    # Table II of Ref. 1 or Eq. 28 of Ref. 2.
    G1 = m_tr2 / (16 * det_um)
    G2 = (m_tr2 - np.trace(M_squared.dot(M_squared))) / (4 * det_um)

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


class TestWeyl:
    """ `tests.synthesis.gate_decompositions.TestWeyl` is the tester class
    for `quick.synthesis.gate_decompositions.two_qubit_decomposition.WeylDecomposition`
    class.
    """
    def test_weyl_coordinates_simple(self) -> None:
        """ Check Weyl coordinates against known basis gates within the Weyl tetrahedron.

        .. math::
            A(a, b, c) = e^{(ia X \otimes X + ib Y \otimes Y + ic Z \otimes Z)}

        Reference for Weyl coordinates, however, we modify the coordinates slightly to match
        the above representation instead:
        https://threeplusone.com/pubs/on_gates.pdf Section 6
        """
        # Identity [0,0,0]
        U = np.identity(4).astype(complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [0, 0, 0], decimal=8)

        # CX [pi/4, 0, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, 0, 0], decimal=8)

        # CY [pi/4, 0, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, 0, 0], decimal=8)

        # CZ [pi/4, 0, 0]
        U = np.array([[
            1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, 0, 0], decimal=8)

        # CH [pi/4, 0, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
            [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, 0, 0], decimal=8)

        # Mølmer–Sørensen [pi/4, 0, 0]
        U = np.array([
            [1, 0, 0, 1j],
            [0, 1, 1j, 0],
            [0, 1j, 1, 0],
            [1j, 0, 0, 1]
        ], dtype=complex) * 1 / np.sqrt(2)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi/4, 0, 0], decimal=8)

        # Magic [pi/4, 0, 0]
        U = np.array([
            [1, 1j, 0, 0],
            [0, 0, 1j, 1],
            [0, 0, 1j, -1],
            [1, -1j, 0, 0]
        ], dtype=complex) * 1 / np.sqrt(2)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, 0, 0], decimal=8)

        # ISWAP (imaginary SWAP) [pi/4, pi/4, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, np.pi / 4, 0], decimal=8)

        # Fermionic SWAP [pi/4, pi/4, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, np.pi / 4, 0], decimal=8)

        # DCX [pi/4, pi/4, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, np.pi / 4, 0], decimal=8)

        # Inverse DCX [pi/4, pi/4, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi / 4, np.pi / 4, 0], decimal=8)

        # SWAP [pi/4, pi/4, pi/4]
        U = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
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

        # Ising XX [t/2, 0, 0]
        t = 0.1
        U = np.array([
            [np.cos(t/2), 0, 0, -1j * np.sin(t/2)],
            [0, np.cos(t/2), -1j * np.sin(t/2), 0],
            [0, -1j * np.sin(t/2), np.cos(t/2), 0],
            [-1j * np.sin(t/2), 0, 0, np.cos(t/2)]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [t/2, 0, 0], decimal=8)

        # Ising YY [t/2, 0, 0]
        t = 0.2
        U = np.array([
            [np.cos(t/2), 0, 0, 1j * np.sin(t/2)],
            [0, np.cos(t/2), -1j * np.sin(t/2), 0],
            [0, -1j * np.sin(t/2), np.cos(t/2), 0],
            [1j * np.sin(t/2), 0, 0, np.cos(t/2)]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [t/2, 0, 0], decimal=8)

        # Ising ZZ [t/2, 0, 0]
        t = 0.3
        U = np.array([
            [np.exp(-1j * t/2), 0, 0, 0],
            [0, np.exp(1j * t/2), 0, 0],
            [0, 0, np.exp(1j * t/2), 0],
            [0, 0, 0, np.exp(-1j * t/2)]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [t/2, 0, 0], decimal=8)

        # CSX [pi/8, 0, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, (1+1j)/2, (1-1j)/2],
            [0, 0, (1-1j)/2, (1+1j)/2]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi/8, 0, 0], decimal=8)

        # XY [t, t, 0]
        t = 0.1
        U = np.array([
            [1, 0, 0, 0],
            [0, np.cos(2*t), -1j * np.sin(2*t), 0],
            [0, -1j * np.sin(2*t), np.cos(2*t), 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [t, t, 0], decimal=8)

        # Givens [t/2, t/2, 0]
        t = 0.1
        U = np.array([
            [1, 0, 0, 0],
            [0, np.cos(t), -np.sin(t), 0],
            [0, np.sin(t), np.cos(t), 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [t/2, t/2, 0], decimal=8)

        # DB [3pi/16, 3pi/16, 0]
        U = np.array([
            [1, 0, 0, 0],
            [0, np.cos(3 * np.pi / 8), -np.sin(3 * np.pi / 8), 0],
            [0, np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [3 * np.pi / 16, 3 * np.pi / 16, 0], decimal=8)

        # SQRT SWAP [pi/8, pi/8, -pi/8]
        U = np.array([
            [1, 0, 0, 0],
            [0, (1 + 1j)/2, (1 - 1j)/2, 0],
            [0, (1 - 1j)/2, (1 + 1j)/2, 0],
            [0, 0, 0, 1]
        ])
        weyl = weyl_coordinates(U)
        assert_almost_equal(weyl, [np.pi/8, np.pi/8, -np.pi/8], decimal=8)

    def test_weyl_coordinates_random(self) -> None:
        """ Randomly check Weyl coordinates with local invariants.
        This test is useful for verifying the correctness of the
        decomposition for arbitrary basis gates, which is useful
        for transpilation if the decomposition supports it.
        """
        for _ in range(30):
            U = unitary_group.rvs(4).astype(complex)
            weyl = weyl_coordinates(U)
            local_equiv = local_equivalence(weyl.astype(float))
            local = two_qubit_local_invariants(U)
            assert_almost_equal(local, local_equiv)