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

__all__ = ["TestStatePreparationShende"]

import copy
import numpy as np
from numpy.testing import assert_almost_equal
import random

from qickit.circuit import QiskitCircuit
from qickit.primitives import Bra, Ket
from qickit.synthesis.statepreparation import Shende
from tests.synthesis.test_statepreparation import StatePreparationTemplate

# Define the test data
generated_data = np.array([random.random() + 1j * random.random() for _ in range(128)])
test_data_bra = Bra(generated_data)
test_data_ket = Ket(generated_data)
checker_data_ket = copy.deepcopy(test_data_ket)
checker_data_bra = copy.deepcopy(test_data_ket.to_bra())


class TestStatePreparationShende(StatePreparationTemplate):
    """ `tests.synthesis.test_statepreparation_shende.TestStatePreparationShende` is the tester class
    for `qickit.synthesis.statepreparation.Shende` class.
    """
    def test_init(self) -> None:
        shende = Shende(QiskitCircuit)

    def test_prepare_state_bra(self) -> None:
        # Initialize the Shende encoder
        shende_encoder = Shende(QiskitCircuit)

        # Encode the data to a circuit
        circuit = shende_encoder.prepare_state(test_data_bra)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_bra.data, decimal=8)

    def test_prepare_state_ket(self) -> None:
        # Initialize the Shende encoder
        shende_encoder = Shende(QiskitCircuit)

        # Encode the data to a circuit
        circuit = shende_encoder.prepare_state(test_data_ket)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_prepare_state_ndarray(self) -> None:
        # Initialize the Shende encoder
        shende_encoder = Shende(QiskitCircuit)

        # Encode the data to a circuit
        circuit = shende_encoder.prepare_state(generated_data)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)