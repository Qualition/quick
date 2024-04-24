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

__all__ = ['test_shende', 'test_mottonen']

import copy
import numpy as np
from numpy.testing import assert_almost_equal
import random

# QICKIT imports
from qickit.data import Data
from qickit.circuit import QiskitCircuit
from qickit.synthesis.statepreparation import Shende, Mottonen


# Test data
generated_data = np.array([random.randint(0, 255) for _ in range(1024)])
test_data = Data(generated_data)
checker_data = copy.deepcopy(test_data)
checker_data.normalize()

def test_shende() -> None:
    """ Test the `qickit.synthesis.statepreparation.Shende` class.
    """
    # Initialize the Shende encoder
    shende_encoder = Shende(QiskitCircuit)

    # Encode the data to a circuit
    circuit = shende_encoder.prepare_state(test_data)

    # Get the state of the circuit
    statevector = circuit.get_statevector()

    # Ensure that the state vector is close enough to the expected state vector
    assert_almost_equal(np.array(statevector), checker_data.data, decimal=8)

def test_mottonen() -> None:
    """ Test the `qickit.synthesis.statepreparation.Mottonen` class.
    """
    # Initialize the Mottonen encoder
    mottonen_encoder = Mottonen(QiskitCircuit)

    # Encode the data to a circuit
    circuit = mottonen_encoder.prepare_state(test_data)

    # Get the state of the circuit
    statevector = circuit.get_statevector()

    # Ensure that the state vector is close enough to the expected state vector
    assert_almost_equal(np.array(statevector), checker_data.data, decimal=8)