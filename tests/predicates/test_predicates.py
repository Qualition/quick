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

__all__ = ["TestPredicates"]

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.stats import unitary_group

from quick.predicates.predicates import (
    is_power,
    is_normalized,
    is_statevector,
    is_square_matrix,
    is_orthogonal_matrix,
    is_real_matrix,
    is_special_matrix,
    is_special_orthogonal_matrix,
    is_special_unitary_matrix,
    is_diagonal_matrix,
    is_symmetric_matrix,
    is_identity_matrix,
    is_unitary_matrix,
    is_hermitian_matrix,
    is_positive_semidefinite_matrix,
    is_isometry,
    is_density_matrix,
    is_product_matrix,
    is_locally_equivalent,
    is_supercontrolled
)


class TestPredicates:
    """ `tests.predicates.TestPredicates` is the tester class for `quick.predicates`
    module.
    """
    def test_is_power(self) -> None:
        """ Test the `is_power()` function.
        """
        assert is_power(2, 2) is True
        assert is_power(3, 2) is False
        assert is_power(2, 4) is True

    def test_is_normalized(self) -> None:
        """ Test the `is_normalized()` function.
        """
        state = np.arange(10).astype(np.complex128)
        assert is_normalized(state) is False
        state = state / np.linalg.norm(state)
        assert is_normalized(state) is True

    @pytest.mark.parametrize("array, system_size, expected", [
        (np.array([1, 0]), 2, True),
        (np.array([0, 1]), 2, True),
        (np.array([1, 0, 0]), 3, True),
        (np.array([1, 2]), 2, False),
        (np.array([1, 2, 3]), 3, False),
        (np.array([1, 0, 0, 0]), 2, True),
        (np.array([[0.5], [0.5], [0.5], [0.5]]), 2, True)
    ])
    def test_is_statevector(
            self,
            array: NDArray[np.complex128],
            system_size: int,
            expected: bool
        ) -> None:
        """ Test the `is_statevector()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a statevector.
        `system_size` : int
            The size of the system. If it's 2, then it's a qubit system.
        `expected` : bool
            The expected output of the function.
        """
        assert is_statevector(array, system_size) is expected

    def test_is_statevector_invalid_system_size(self) -> None:
        """ Test the `is_statevector()` function with invalid system size.
        """
        with pytest.raises(ValueError):
            is_statevector(np.array([1, 0]), system_size=0)

    @pytest.mark.parametrize("array, expected", [
        (np.random.rand(2, 2), True),
        (np.random.rand(3, 3), True),
        (np.random.rand(4, 4), True),
        (np.random.rand(5, 5), True),
        (np.random.rand(2, 1), False),
        (np.random.rand(1, 3), False),
        (np.random.rand(4, 12), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_square_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_square_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a square matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_square_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [1, 0],
            [0, 1]
        ]), True),
        (np.array([
            [0, 1],
            [1, 0]
        ]), True),
        (np.array([
            [1, 2],
            [3, 4]
        ]), False)
    ])
    def test_is_orthogonal_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_orthogonal_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is an orthogonal matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_orthogonal_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([1, 2, 3]), True),
        (np.array([1+1j, 2, 3]), False)
    ])
    def test_is_real_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_real_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a real matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_real_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [1, 0],
            [0, 1]
        ]), True),
        (np.array([
            [0, -1],
            [1, 0]
        ]), True),
        (np.array([
            [1, 1],
            [1, 1]
        ]), False)
    ])
    def test_is_special_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_special_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a special matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_special_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [-0.24665563, 0.52914357, -0.3101837, 0.7503027],
            [-0.33006535, 0.74061563, 0.27071591, -0.51890099],
            [-0.6150982, -0.31194876, 0.66273065, 0.2917709],
            [-0.6722143, -0.27236654, -0.62552942, -0.28750192]
        ]), True),
        (np.array([
            [0.0181244, 0.8330495 , 0.55219219, -0.02799671],
            [0.8772641, -0.06010757, 0.03781642, -0.47472592],
            [0.1060249, 0.5481764 , -0.82739822, 0.0606098],
            [0.46780116, -0.04379778, 0.09521496, 0.87759782]
        ]), True),
        (np.array([
            [1, 2, 3],
            [3, 4, 5]
        ]), False),
        (np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]), False)
    ])
    def test_is_special_orthogonal_matrix(
            self,
            array: NDArray[np.float64],
            expected: bool
        ) -> None:
        """ Test the `is_special_orthogonal_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.float64]
            The input array to check if it is a special orthogonal matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_special_orthogonal_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [0.38422631+2.83471478e-01j, 0.25943494+7.65221683e-02j, 0.56281274+3.91037228e-02j, 0.54010984-2.98070487e-01j],
            [0.34951286-5.51982645e-01j, -0.1755857 +1.77050098e-01j, 0.38094516+5.26882445e-01j, -0.28000149+9.92657401e-02j],
            [0.01016025+4.69180869e-01j, 0.05238197-4.75213843e-01j, 0.44501419+5.84109815e-03j, -0.54594179+2.34669613e-01j],
            [0.17806077+3.05336581e-01j, 0.1054005 +7.90556427e-01j, -0.00897634-2.46649688e-01j, -0.42196931+6.80411917e-04j]
        ]), True),
        (np.array([
            [0.05270704-0.35904018j, -0.54492897-0.41275395j, -0.14347842-0.07950459j, -0.23602791-0.56425393j],
            [0.19398055+0.62964354j, -0.18457864+0.15122235j, -0.61785577-0.32821084j, 0.04745637-0.13138828j],
            [0.64639778+0.04023874j, -0.3498751 +0.00456368j, 0.43706868-0.33284833j, -0.14751464+0.36679658j],
            [0.12091386-0.01277784j, 0.43253418+0.407713j, 0.26579003-0.33341187j, -0.35797256-0.56740522j]
        ]), False)
    ])
    def test_is_special_unitary_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_special_unitary_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a special unitary matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_special_unitary_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.diag([1, 2, 3]), True),
        (np.diag([4, 5, 6, 7]), True),
        (np.diag([8, 9]), True),
        (np.array([
            [1, 2, 0],
            [0, 3, 4],
            [0, 0, 5]
        ]), False),
        (np.array([
            [1, 0, 0],
            [2, 3, 0],
            [0, 0, 4]
        ]), False),
        (np.array([
            [1, 0],
            [1, 1]
        ]), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_diagonal_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_diagonal_matrix()` function with diagonal matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a diagonal matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_diagonal_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]
        ]), True),
        (np.array([
            [1, 2, 3, 4],
            [2, 5, 6, 7],
            [3, 6, 8, 9],
            [4, 7, 9, 10]
        ]), True),
        (np.array([
            [1, 2],
            [2, 3]
        ]), True),
        (np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]), False),
        (np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]), False),
        (np.array([
            [1, 2],
            [3, 4]
        ]), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_symmetric_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_symmetric_matrix()` function with symmetric matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a symmetric matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_symmetric_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.eye(2), True),
        (np.eye(3), True),
        (np.eye(4), True),
        (np.eye(5), True),
        (np.random.rand(2, 2), False),
        (np.random.rand(3, 3), False),
        (np.random.rand(4, 4), False),
        (np.random.rand(3, 4), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_identity_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_identity_matrix()` function with identity matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is an identity matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_identity_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (unitary_group.rvs(2), True),
        (unitary_group.rvs(3), True),
        (unitary_group.rvs(4), True),
        (unitary_group.rvs(5), True),
        (np.random.rand(2, 2), False),
        (np.random.rand(3, 3), False),
        (np.random.rand(3, 4), False),
        (np.random.rand(5, 2), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_unitary_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_unitary_matrix()` function.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a unitary matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_unitary_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [1, 2 + 1j, 3],
            [2 - 1j, 4, 5 + 2j],
            [3, 5 - 2j, 6]
        ]), True),
        (np.array([
            [1, 2 + 1j],
            [2 - 1j, 3]
        ]), True),
        (np.array([
            [1, 0],
            [0, 1]
        ]), True),
        (np.array([
            [1, 2 + 1j, 3],
            [2 + 1j, 4, 5 + 2j],
            [3, 5 - 2j, 6]
        ]), False),
        (np.array([
            [1, 2 + 1j],
            [2 + 1j, 3]
        ]), False),
        (np.array([
            [1, 2],
            [3, 4]
        ]), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_hermitian_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_hermitian_matrix()` function with Hermitian matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a Hermitian matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_hermitian_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [1, 0],
            [0, 1]
        ]), True),
        (np.array([
            [1, 0],
            [0, 0]
        ]), True),
        (np.array([
            [1, 2],
            [2, 3]
        ]), False),
        (np.array([
            [1, 2],
            [3, 4]
        ]), False),
        (np.array([
            [1, 0],
            [0, -1]
        ]), False),
        (np.array([
            [1, 2],
            [2, 1]
        ]), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_positive_semidefinite_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_positive_semidefinite_matrix()` function with
        positive semidefinite matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is a positive semidefinite matrix.
        `expected` : bool
            The expected output of the function.
        """
        assert is_positive_semidefinite_matrix(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [0.56078693+0.13052803j, -0.31583062-0.08879493j],
            [0.7123732 +0.2419316j, 0.39227097-0.22401521j],
            [0.30203025+0.10607406j, -0.38000351+0.7374988j]
        ], dtype=np.complex128), True),
        (np.array([
            [0.08849653+0.24435482j],
            [-0.72166734+0.64160373j]
        ], dtype=np.complex128), True),
        (np.array([
            [0.02572621+0.08711405j, -0.84637795+0.52477964j],
            [0.86175428-0.4991281j, -0.06470557-0.06374861j]
        ]), True),
        (np.array([
            [1, 1],
            [1, 1]
        ], dtype=np.complex128), False),
        (np.random.rand(3, 3), False),
        (np.random.rand(1, 2), False),
        (np.array([1, 0]), False),
    ])
    def test_is_isometry(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_isometry` function with various matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is an isometry.
        `expected` : bool
            The expected output of the function.
        """
        assert is_isometry(array) is expected

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [0.55282636+0.j, 0.19339888+0.1369917j],
            [0.19339888-0.1369917j, 0.44717364+0.j]
        ], dtype=np.complex128), True),
        (np.array([
            [0.19834677+0.j, 0.21077084+0.08851485j, 0.07369894-0.03780167j],
            [0.21077084-0.08851485j, 0.5556912 +0.j, 0.09721694-0.13294078j],
            [0.07369894+0.03780167j, 0.09721694+0.13294078j, 0.24596203+0.j]
        ], dtype=np.complex128), True),
        (np.array([
            [1, 2 + 1j],
            [2 + 1j, 3]
        ]), False),
        (np.array([
            [1, 2],
            [3, 4]
        ]), False),
        (np.random.rand(2, 3, 3), False)
    ])
    def test_is_density_matrix(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_density_matrix` function with various matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is an isometry.
        `expected` : bool
            The expected out of the function.
        """
        assert is_density_matrix(array) is expected

    def test_is_product_matrix(self) -> None:
        """ Test the `is_product_matrix` function.
        """
        u1 = unitary_group.rvs(2)
        u2 = unitary_group.rvs(2)
        u3 = np.kron(u1, u2).astype(complex)
        assert is_product_matrix(u3) is True
        cx = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        assert is_product_matrix(cx) is False

    def test_is_locally_equivalent(self) -> None:
        """ Test the `is_locally_equivalent` function.
        """
        cx = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ])
        cz = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        assert is_locally_equivalent(cx, cz) is True

        swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        assert is_locally_equivalent(cx, swap) is False

    @pytest.mark.parametrize("array, expected", [
        (np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=complex), True),
        (np.array([
            [1, 0, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1, 0],
            [0, 1j, 0, 0]
        ], dtype=complex), True),
        (np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex), True),
        (np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex), True),
        (np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex), False),
        (np.array([
            [1, 0, 0, 0],
            [0, 0.5+0.5j, 0, 0.5-0.5j],
            [0, 0, 1, 0],
            [0, 0.5-0.5j, 0, 0.5+0.5j]
        ], dtype=complex), False)
    ])
    def test_is_supercontrolled(
            self,
            array: NDArray[np.complex128],
            expected: bool
        ) -> None:
        """ Test the `is_supercontrolled` function with various matrices.

        Parameters
        ----------
        `array` : NDArray[np.complex128]
            The input array to check if it is supercontrolled.
        `expected` : bool
            The expected output of the function.
        """
        assert is_supercontrolled(array) is expected