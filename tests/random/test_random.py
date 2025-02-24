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

from quick.predicates import is_unitary_matrix
from quick.random import generate_random_state, generate_random_unitary


class TestRandom:
    """ `tests.random.TestRandom` is the tester class for `quick.random`
    module.
    """
    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_state(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_state` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the state vector.
        """
        state = generate_random_state(num_qubits)

        assert state.shape == (2 ** num_qubits,)
        assert pytest.approx(1.0) == abs(state @ state.conj())

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_generate_random_unitary(
            self,
            num_qubits: int
        ) -> None:
        """ Test the `generate_random_unitary` function.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits in the unitary matrix.
        """
        unitary = generate_random_unitary(num_qubits)

        assert unitary.shape == (2 ** num_qubits, 2 ** num_qubits)
        assert is_unitary_matrix(unitary)