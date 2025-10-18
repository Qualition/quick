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

__all__ = ["TestRandom"]

import pytest

from quick.predicates import (
    is_unitary_matrix,
    is_statevector,
    is_density_matrix,
    is_orthogonal_matrix,
    is_special_orthogonal_matrix,
    is_special_unitary_matrix
)
from quick.random import (
    generate_random_state,
    generate_random_unitary,
    generate_random_density_matrix,
    generate_random_orthogonal_matrix,
    generate_random_special_orthogonal_matrix,
    generate_random_special_unitary_matrix
)


class TestRandom:
    """ `tests.random.TestRandom` is the tester class for `quick.random`
    module.
    """
    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_state(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_state()` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the state vector.
        """
        state = generate_random_state(num_qubits)

        assert is_statevector(state)

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_unitary(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_unitary()` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the unitary matrix.
        """
        unitary = generate_random_unitary(num_qubits)

        assert unitary.shape == (2 ** num_qubits, 2 ** num_qubits)
        assert is_unitary_matrix(unitary)

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("generator", ["hilbert-schmidt", "bures"])
    @pytest.mark.parametrize("rank", [1, 2, 3, None])
    def test_generate_random_density_matrix(
            self,
            num_qubits: int,
            generator: str,
            rank: int
        ) -> None:
        """ Test the `generate_random_density_matrix()` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the density matrix.
        `generator` : str
            The method to use for generating the density matrix.
        `rank` : int
            The rank of the density matrix.
        """
        density_matrix = generate_random_density_matrix(
            num_qubits=num_qubits,
            rank=rank,
            generator=generator # type: ignore
        )

        assert is_density_matrix(density_matrix)

    def test_generate_random_density_matrix_invalid_generator(
            self
        ) -> None:
        """ Test the `generate_random_density_matrix()` function with an
        invalid generator.
        """
        with pytest.raises(ValueError):
            generate_random_density_matrix(
                num_qubits=2,
                rank=1,
                generator="invalid-generator" # type: ignore
            )

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_orthogonal_matrix(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_orthogonal_matrix()` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the SO matrix.
        """
        o_matrix = generate_random_orthogonal_matrix(num_qubits)

        assert is_orthogonal_matrix(o_matrix)

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_special_orthogonal_matrix(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_special_orthogonal_matrix()` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the SO matrix.
        """
        so_matrix = generate_random_special_orthogonal_matrix(num_qubits)

        assert is_special_orthogonal_matrix(so_matrix)

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_special_unitary_matrix(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_special_unitary_matrix()` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the SU matrix.
        """
        su_matrix = generate_random_special_unitary_matrix(num_qubits)

        assert is_special_unitary_matrix(su_matrix)