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

__all__ = ["TestMCXVChain"]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
import pytest

from quick.circuit import QiskitCircuit
from quick.synthesis.gate_decompositions.multi_controlled_decomposition.mcx_vchain import MCXVChain


# Folder prefix
folder_prefix = "tests/synthesis/gate_decompositions/multi_controlled_decomposition/"

# Define the expected values
mcx_vchain_4 = np.load(folder_prefix + "mcx_vchain_4.npy")
mcx_vchain_5 = np.load(folder_prefix + "mcx_vchain_5.npy")
mcx_vchain_6 = np.load(folder_prefix + "mcx_vchain_6.npy")


class TestMCXVChain:
    @pytest.mark.parametrize("num_controls", [2, 3, 4, 5, 6])
    def test_get_num_ancillas(self, num_controls: int) -> None:
        """ Test the `MCXVChain.get_num_ancillas()`.

        Parameters
        ----------
        `num_controls` : int
            The number of control qubits.
        """
        assert MCXVChain.get_num_ancillas(num_controls) == max(0, num_controls - 2)

    @pytest.mark.parametrize("num_controls, expected", [
        [4, mcx_vchain_4],
        [5, mcx_vchain_5],
        [6, mcx_vchain_6],
    ])
    def test_define_decomposition(
            self,
            num_controls: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `MCXVChain.define_decomposition()`.

        Parameters
        ----------
        `num_controls` : int
            The number of control qubits.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        mcx_vchain_decomposition = MCXVChain()

        assert_almost_equal(
            mcx_vchain_decomposition.define_decomposition(
                num_controls,
                QiskitCircuit
            ).get_unitary(),
            expected,
            8
        )

    @pytest.mark.parametrize("num_controls, expected", [
        [4, mcx_vchain_4],
        [5, mcx_vchain_5],
        [6, mcx_vchain_6],
    ])
    def test_apply_decomposition(
            self,
            num_controls: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `MCXVChain.apply_decomposition()`.

        Parameters
        ----------
        `num_controls` : int
            The number of control qubits.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        mcx_vchain_decomposition = MCXVChain()

        qubits = list(range(num_controls + mcx_vchain_decomposition.get_num_ancillas(num_controls) + 1))

        circuit = QiskitCircuit(num_controls + mcx_vchain_decomposition.get_num_ancillas(num_controls) + 1)
        mcx_vchain_decomposition.apply_decomposition(
            circuit,
            qubits[:num_controls],
            qubits[num_controls],
            qubits[num_controls + 1:]
        )

        assert_almost_equal(
            circuit.get_unitary(),
            expected,
        )