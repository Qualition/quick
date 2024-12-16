# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/QICKIT/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

__all__ = ["test_mcx_vchain"]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
import pytest

from qickit.circuit import QiskitCircuit
from qickit.synthesis.gate_decompositions.multi_controlled_decomposition.mcx_vchain import mcx_vchain_decomposition


# Folder prefix
folder_prefix = "tests/synthesis/gate_decompositions/multi_controlled_decomposition/"

# Define the expected values
mcx_vchain_4 = np.load(folder_prefix + "mcx_vchain_4.npy")
mcx_vchain_5 = np.load(folder_prefix + "mcx_vchain_5.npy")
mcx_vchain_6 = np.load(folder_prefix + "mcx_vchain_6.npy")


@pytest.mark.parametrize("num_controls, expected", [
    [4, mcx_vchain_4],
    [5, mcx_vchain_5],
    [6, mcx_vchain_6],
])
def test_mcx_vchain(
        num_controls: int,
        expected: NDArray[np.complex128]
    ) -> None:
    """ Test the MCX V-chain decomposition.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.
    `expected` : NDArray[np.complex128]
        The expected unitary matrix.
    """
    assert_almost_equal(
        mcx_vchain_decomposition(num_controls, QiskitCircuit).get_unitary(),
        expected,
        8
    )