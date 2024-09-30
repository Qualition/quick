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

unitary_matrix = unitary_group.rvs(8).astype(complex)


class TestShendeCompiler(Template):
    """ `tests.compiler.TestShendeCompiler` is the tester for the `qickit.compiler.Compiler` class.
    """
    def test_init(self) -> None:
        Compiler(circuit_framework=TKETCircuit)

    def test_init_invalid_framework(self) -> None:
        with pytest.raises(TypeError):
            Compiler(circuit_framework=int) # type: ignore

    def test_init_invalid_state_preparation(self) -> None:
        with pytest.raises(TypeError):
            Compiler(circuit_framework=TKETCircuit, state_preparation=int) # type: ignore

    def test_init_invalid_unitary_preparation(self) -> None:
        with pytest.raises(TypeError):
            Compiler(circuit_framework=TKETCircuit, unitary_preparation=int) # type: ignore

    def test_init_invalid_optimizer(self) -> None:
        with pytest.raises(TypeError):
            Compiler(circuit_framework=TKETCircuit, optimizer=0) # type: ignore

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
        circuit = shende_compiler.unitary_preparation(unitary_matrix)

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_optimize(self) -> None:
        # TODO: Implement the test_optimize method
        pass

    def test_check_primitive(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the checker does not raise an error when a valid primitive is passed
        shende_compiler._check_primitive(test_data_ket)
        shende_compiler._check_primitive(test_data_bra)
        shende_compiler._check_primitive(Operator(unitary_matrix))
        shende_compiler._check_primitive(generated_statevector)
        shende_compiler._check_primitive(unitary_matrix)

    def test_check_primitive_invalid_primitive(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the checker raises a ValueError when an invalid primitive is passed
        with pytest.raises(TypeError):
            shende_compiler._check_primitive(0) # type: ignore

        with pytest.raises(TypeError):
            shende_compiler._check_primitive([0]) # type: ignore

        with pytest.raises(ValueError):
            shende_compiler._check_primitive(np.array([0]))

        with pytest.raises(ValueError):
            shende_compiler._check_primitive(np.array(np.zeros((2, 2, 2), dtype=complex)))

        with pytest.raises(ValueError):
            shende_compiler._check_primitive(np.array(np.zeros((2, 3), dtype=complex)))

    def test_check_primitive_qubits(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the checker does not raise an error when valid qubits are passed
        shende_compiler._check_primitive_qubits(test_data_ket, range(7))
        shende_compiler._check_primitive_qubits(test_data_bra, range(7))
        shende_compiler._check_primitive_qubits(Operator(unitary_matrix), range(3))
        shende_compiler._check_primitive_qubits(generated_statevector, range(7))
        shende_compiler._check_primitive_qubits(unitary_matrix, range(3))

    def test_check_primitive_qubits_invalid_qubits(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the checker raises a ValueError when invalid qubits are passed
        with pytest.raises(ValueError):
            shende_compiler._check_primitive_qubits(test_data_ket, range(8))

    def test_check_primitives(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the checker does not raise an error when valid primitives are passed
        shende_compiler._check_primitives([
            (test_data_ket, range(7)),
            (test_data_bra, range(7)),
            (Operator(unitary_matrix), range(3)),
            (generated_statevector, range(7)),
            (unitary_matrix, range(3))
        ])

    def test_compile_primitive_bra(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler._compile_primitive(test_data_bra)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_bra.data, decimal=8)


    def test_compile_primitive_ket(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler._compile_primitive(test_data_ket)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

    def test_compile_primitive_operator(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler._compile_primitive(Operator(unitary_matrix))

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_compile_primitive_ndarray(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Encode the data to a circuit
        circuit = shende_compiler._compile_primitive(generated_statevector)

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        assert_almost_equal(np.array(statevector), checker_data_ket.data.flatten(), decimal=8)

        # Encode the data to a circuit
        circuit = shende_compiler._compile_primitive(unitary_matrix)

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_compile_primitive_invalid_primitive(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Ensure that the compiler raises a ValueError when an invalid primitive is passed
        with pytest.raises(TypeError):
            shende_compiler._compile_primitive(0) # type: ignore

        with pytest.raises(TypeError):
            shende_compiler._compile_primitive([0]) # type: ignore

        with pytest.raises(ValueError):
            shende_compiler._compile_primitive(np.array([0]))

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
        circuit = shende_compiler.compile(Operator(unitary_matrix))

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
        circuit = shende_compiler.compile(unitary_matrix)

        # Get the unitary matrix of the circuit
        unitary = circuit.get_unitary()

        # Ensure that the unitary matrix is close enough to the expected unitary matrix
        assert_almost_equal(unitary, unitary_matrix, decimal=8)

    def test_compile_multiple_primitives(self) -> None:
        # Initialize the Shende compiler
        shende_compiler = Compiler(circuit_framework=TKETCircuit)

        # Generate a random bra and ket over three qubits
        generated_statevector = np.array([random.random() + 1j * random.random() for _ in range(8)])
        test_data_ket = Ket(generated_statevector)
        test_data_bra = Bra(generated_statevector)
        checker_data_ket = copy.deepcopy(test_data_ket)
        checker_data_bra = copy.deepcopy(test_data_ket.to_bra())

        # Generate two random unitary matrix over three qubits
        unitary_matrix_1 = unitary_group.rvs(8).astype(complex)
        unitary_matrix_2 = unitary_group.rvs(8).astype(complex)

        # Encode the data to a circuit
        circuit = shende_compiler.compile([
            (test_data_ket, [0, 1, 2]),
            (test_data_bra, [3, 4, 5]),
            (unitary_matrix_1, [0, 1, 2]),
            (unitary_matrix_2, [3, 4, 5])
        ])

        # TODO: Add a check for the optimized circuit

        # Get the state of the circuit
        statevector = circuit.get_statevector()

        # Ensure that the state vector is close enough to the expected state vector
        checker_data_ket = np.dot(unitary_matrix_1, checker_data_ket.data.flatten())
        checker_data_bra = np.dot(unitary_matrix_2, checker_data_bra.data.flatten())
        checker_statevector = np.kron(checker_data_bra, checker_data_ket)

        assert_almost_equal(np.array(statevector), checker_statevector, decimal=8)