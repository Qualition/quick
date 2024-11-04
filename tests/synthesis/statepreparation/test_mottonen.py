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

__all__ = ["TestStatePreparationMottonen"]

import copy
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import random

from qickit.circuit import QiskitCircuit
from qickit.primitives import Bra, Ket
from qickit.synthesis.statepreparation import Mottonen
from tests.synthesis.statepreparation import StatePreparationTemplate

# Define the test data
generated_data = np.array([random.random() + 1j * random.random() for _ in range(128)])
test_data_bra = Bra(generated_data)
test_data_ket = Ket(generated_data)
checker_data_ket = copy.deepcopy(test_data_ket)
checker_data_bra = copy.deepcopy(test_data_ket.to_bra())


class TestStatePreparationMottonen(StatePreparationTemplate):
    """ `tests.synthesis.test_mottonen.TestStatePreparationMottonen` is the tester class
    for `qickit.synthesis.statepreparation.Mottonen` class.
    """
    def test_init(self) -> None:
        Mottonen(QiskitCircuit)

    def test_init_invalid_output_framework(self) -> None:
        with pytest.raises(TypeError):
            Mottonen("invalid_framework") # type: ignore

    def test_prepare_state_bra(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Encode the data to a circuit
        circuit = shende_encoder.prepare_state(test_data_bra)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_bra.data, decimal=8)

    def test_prepare_state_ket(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Encode the data to a circuit
        circuit = shende_encoder.prepare_state(test_data_ket)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_prepare_state_ndarray(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Encode the data to a circuit
        circuit = shende_encoder.prepare_state(generated_data)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_apply_state_ket(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Initialize the circuit
        circuit = QiskitCircuit(7)

        # Apply the state to a circuit
        circuit = shende_encoder.apply_state(circuit, test_data_ket, range(7))

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_apply_state_bra(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Initialize the circuit
        circuit = QiskitCircuit(7)

        # Apply the state to a circuit
        circuit = shende_encoder.apply_state(circuit, test_data_bra, range(7))

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_bra.data, decimal=8)

    def test_apply_state_ndarray(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Initialize the circuit
        circuit = QiskitCircuit(7)

        # Apply the state to a circuit
        circuit = shende_encoder.apply_state(circuit, generated_data, range(7))

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_apply_state_invalid_input(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Initialize the circuit
        circuit = QiskitCircuit(7)

        # Apply the state to a circuit
        with pytest.raises(TypeError):
            shende_encoder.apply_state(circuit, "invalid_input", range(7)) # type: ignore

    def test_apply_state_invalid_qubit_indices(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Initialize the circuit
        circuit = QiskitCircuit(7)

        # Apply the state to a circuit
        with pytest.raises(TypeError):
            shende_encoder.apply_state(circuit, test_data_ket, "invalid_qubit_indices") # type: ignore

        with pytest.raises(TypeError):
            shende_encoder.apply_state(circuit, test_data_ket, [1+1j, 2+2j, 3+3j]) # type: ignore

    def test_apply_state_qubit_indices_out_of_range(self) -> None:
        # Initialize the Mottonen encoder
        shende_encoder = Mottonen(QiskitCircuit)

        # Initialize the circuit
        circuit = QiskitCircuit(7)

        # Apply the state to a circuit
        with pytest.raises(IndexError):
            shende_encoder.apply_state(circuit, test_data_ket, [0, 1, 2, 3, 4, 5, 12])