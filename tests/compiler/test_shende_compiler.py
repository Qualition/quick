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

__all__ = ["TestShendeCompiler"]

import copy
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import random
from scipy.stats import unitary_group

from qickit.circuit import TKETCircuit
from qickit.compiler import Compiler
from qickit.primitives import Bra, Ket, Operator

from tests.compiler import Template

# Define the test data
generated_statevector = np.array([random.random() + 1j * random.random() for _ in range(128)])
test_data_bra = Bra(generated_statevector)
test_data_ket = Ket(generated_statevector)
checker_data_ket = copy.deepcopy(test_data_ket)
checker_data_bra = copy.deepcopy(test_data_ket.to_bra())

unitary_matrix = unitary_group.rvs(8)


class TestShendeCompiler(Template):
    """ `tests.compiler.TestShendeCompiler` is the tester for the `qickit.compiler.Compiler` class.
    """
    def test_init(self) -> None:
        Compiler(circuit_framework=TKETCircuit)

    def test_state_preparation(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler.state_preparation(generated_statevector)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_unitary_preparation(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler.unitary_preparation(unitary_matrix) # type: ignore

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_compile_bra(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler.compile(test_data_bra)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_bra.data, decimal=8)

    def test_compile_ket(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler.compile(test_data_ket)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_compile_operator(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler.compile(Operator(unitary_matrix)) # type: ignore

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_compile_ndarray(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler.compile(generated_statevector)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

        # Encode the data to a circuit
        circuit = shende_compiler.compile(unitary_matrix) # type: ignore

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_compile_invalid_primitive(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the compiler raises a ValueError when an invalid primitive is passed
        with pytest.raises(ValueError):
            shende_compiler.compile(0) # type: ignore