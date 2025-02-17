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

__all__ = ["TestMCXUtils"]

import numpy as np
from numpy.testing import assert_almost_equal

from quick.circuit import QiskitCircuit
from quick.synthesis.gate_decompositions.multi_controlled_decomposition.mcx_utils import (
    CCX, RCCX, C3X, C3SX, RC3X, C4X
)

# Folder prefix
folder_prefix = "tests/synthesis/gate_decompositions/multi_controlled_decomposition/"

ccx_unitary = np.load(folder_prefix + "ccx.npy")
c3x_unitary = np.load(folder_prefix + "c3x.npy")
c3sx_unitary = np.load(folder_prefix + "c3sx.npy")
c4x_unitary = np.load(folder_prefix + "c4x.npy")
rccx_unitary = np.load(folder_prefix + "rccx.npy")
rc3x_unitary = np.load(folder_prefix + "rc3x.npy")


class TestMCXUtils:
    """ `tests.synthesis.gate_decompositions.TestMCXUtils` is the tester class for
    `quick.synthesis.gate_decompositions.multi_controlled_decomposition.mcx_utils`
    module.
    """
    def test_CCX(self) -> None:
        """ Test the CCX decomposition.
        """
        circuit = QiskitCircuit(3)
        CCX(circuit, [0, 1], 2)
        assert_almost_equal(circuit.get_unitary(), ccx_unitary, 8)

    def test_RCCX(self) -> None:
        """ Test the RCCX decomposition.
        """
        circuit = QiskitCircuit(3)
        RCCX(circuit, [0, 1], 2)
        assert_almost_equal(circuit.get_unitary(), rccx_unitary, 8)

    def test_C3X(self) -> None:
        """ Test the C3X decomposition.
        """
        circuit = QiskitCircuit(4)
        C3X(circuit, [0, 1, 2], 3)
        assert_almost_equal(circuit.get_unitary(), c3x_unitary, 8)

    def test_C3SX(self) -> None:
        """ Test the C3SX decomposition.
        """
        circuit = QiskitCircuit(4)
        C3SX(circuit, [0, 1, 2], 3)
        assert_almost_equal(circuit.get_unitary(), c3sx_unitary, 8)

    def test_RC3X(self) -> None:
        """ Test the RC3X decomposition.
        """
        circuit = QiskitCircuit(4)
        RC3X(circuit, [0, 1, 2], 3)
        assert_almost_equal(circuit.get_unitary(), rc3x_unitary, 8)

    def test_C4X(self) -> None:
        """ Test the C4X decomposition.
        """
        circuit = QiskitCircuit(5)
        C4X(circuit, [0, 1, 2, 3], 4)
        assert_almost_equal(circuit.get_unitary(), c4x_unitary, 8)