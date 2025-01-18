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

__all__ = ["TestDiffusion"]

from numpy.testing import assert_almost_equal
import pytest
import random
from typing import Type

from quick.circuit import Circuit, QiskitCircuit
from quick.primitives import Operator
from quick.synthesis.unitarypreparation import Diffusion
from tests.synthesis.unitarypreparation import UnitaryPreparationTemplate

# Define the test data
def generate_random_circuit(
        max_depth: int,
        qc_framework: Type[Circuit]
    ) -> Circuit:
    """ Generate a random circuit using the allowed gate set
    with a given maximum depth.

    Parameters
    ----------
    `max_depth` : int
        Maximum depth of the circuit.
    `qc_framework` : type[quick.circuit.Circuit]
        Quantum circuit framework to be used.

    Returns
    -------
    `circuit` : quick.circuit.Circuit
        Random circuit.
    """
    n_qubits = 3
    circuit = qc_framework(n_qubits)

    gates = ["h", "cx", "ccx", "swap", "z"]

    for _ in range(max_depth):
        gate = random.choice(gates)

        match gate:
            case "h":
                qubit = random.randint(0, n_qubits - 1)
                circuit.H(qubit)
            case "cx":
                control_qubit = random.randint(0, n_qubits - 1)
                target_qubit = random.randint(0, n_qubits - 1)
                while target_qubit == control_qubit:
                    target_qubit = random.randint(0, n_qubits - 1)
                circuit.CX(control_qubit, target_qubit)
            case "ccx":
                control_qubit1 = random.randint(0, n_qubits - 1)
                control_qubit2 = random.randint(0, n_qubits - 1)
                target_qubit = random.randint(0, n_qubits - 1)
                while control_qubit2 == control_qubit1:
                    control_qubit2 = random.randint(0, n_qubits - 1)
                while target_qubit == control_qubit1 or target_qubit == control_qubit2:
                    target_qubit = random.randint(0, n_qubits - 1)
                circuit.MCX([control_qubit1, control_qubit2], target_qubit)
            case "swap":
                qubit1 = random.randint(0, n_qubits - 1)
                qubit2 = random.randint(0, n_qubits - 1)
                while qubit2 == qubit1:
                    qubit2 = random.randint(0, n_qubits - 1)
                circuit.SWAP(qubit1, qubit2)
            case "z":
                qubit = random.randint(0, n_qubits - 1)
                circuit.Z(qubit)

    return circuit

test_circuit = generate_random_circuit(7, QiskitCircuit)
test_unitary = test_circuit.get_unitary()


@pytest.mark.slow
class TestDiffusion(UnitaryPreparationTemplate):
    """ `tests.synthesis.test_diffusion.TestDiffusion` is the tester class
    for `quick.synthesis.unitarypreparation.Diffusion` class.
    """
    def test_init(self) -> None:
        Diffusion(QiskitCircuit)

    def test_init_invalid_output_framework(self) -> None:
        with pytest.raises(TypeError):
            Diffusion("invalid output framework") # type: ignore

    def test_prepare_unitary_ndarray(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        for _ in range(10):
            test_circuit = generate_random_circuit(7, QiskitCircuit)
            test_unitary = test_circuit.get_unitary()

            # Prepare the unitary matrix
            circuit = diffusion_model.prepare_unitary(test_unitary) # type: ignore

            # Get the unitary matrix of the circuit
            unitary = circuit.get_unitary()

            # Ensure that the unitary matrix is close enough to the expected unitary matrix
            assert_almost_equal(unitary, test_unitary, decimal=8)

    def test_prepare_unitary_operator(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        for _ in range(10):
            test_circuit = generate_random_circuit(7, QiskitCircuit)
            test_unitary = test_circuit.get_unitary()

            # Prepare the unitary matrix
            circuit = diffusion_model.prepare_unitary(Operator(test_unitary)) # type: ignore

            # Get the unitary matrix of the circuit
            unitary = circuit.get_unitary()

            # Ensure that the unitary matrix is close enough to the expected unitary matrix
            assert_almost_equal(unitary, test_unitary, decimal=8)

    def test_apply_unitary_ndarray(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        for _ in range(10):
            test_circuit = generate_random_circuit(7, QiskitCircuit)
            test_unitary = test_circuit.get_unitary()

            # Initialize the quick circuit
            circuit = QiskitCircuit(3)

            # Apply the unitary matrix to the circuit
            circuit = diffusion_model.apply_unitary(circuit, test_unitary, range(3))

            # Get the unitary matrix of the circuit
            unitary = circuit.get_unitary()

            # Ensure that the unitary matrix is close enough to the expected unitary matrix
            assert_almost_equal(unitary, test_unitary, decimal=8)

    def test_apply_unitary_operator(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        for _ in range(10):
            test_circuit = generate_random_circuit(7, QiskitCircuit)
            test_unitary = test_circuit.get_unitary()

            # Initialize the quick circuit
            circuit = QiskitCircuit(3)

            # Apply the unitary matrix to the circuit
            circuit = diffusion_model.apply_unitary(circuit, Operator(test_unitary), range(3))

            # Get the unitary matrix of the circuit
            unitary = circuit.get_unitary()

            # Ensure that the unitary matrix is close enough to the expected unitary matrix
            assert_almost_equal(unitary, test_unitary, decimal=8)

    def test_apply_unitary_invalid_input(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        # Initialize the quick circuit
        circuit = QiskitCircuit(3)

        with pytest.raises(TypeError):
            diffusion_model.apply_unitary(circuit, "invalid input", range(3)) # type: ignore

    def test_apply_unitary_invalid_qubit_indices(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        # Initialize the quick circuit
        circuit = QiskitCircuit(3)

        with pytest.raises(TypeError):
            diffusion_model.apply_unitary(circuit, test_unitary, "invalid qubit indices") # type: ignore

        with pytest.raises(TypeError):
            diffusion_model.apply_unitary(circuit, test_unitary, [1+1j, 2+2j, 3+3j]) # type: ignore

        with pytest.raises(ValueError):
            diffusion_model.apply_unitary(circuit, test_unitary, [0, 1, 2, 3])

    def test_apply_unitary_invalid_qubit_indices_out_of_range(self) -> None:
        # Initialize the Diffusion
        diffusion_model = Diffusion(QiskitCircuit)

        # Initialize the quick circuit
        circuit = QiskitCircuit(3)

        with pytest.raises(IndexError):
            diffusion_model.apply_unitary(circuit, test_unitary, [0, 1, 4])