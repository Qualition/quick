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

__all__ = ["TestCircuitBase"]

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from typing import Type

from quick.circuit import Circuit
from quick.random import generate_random_state, generate_random_unitary

from tests.circuit import CIRCUIT_FRAMEWORKS


class TestCircuitBase:
    """ `tests.circuit.TestCircuitBase` class is a tester for the base functionality
    of the `quick.circuit.Circuit` class.
    """
    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_init(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the initialization of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        circuit_framework(2)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_num_qubits_value_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test to see if the error is raised when the number of qubits
        is less than or equal to 0.
        """
        # Ensure the error is raised when the number of qubits is less than or equal to 0
        with pytest.raises(ValueError):
            circuit_framework(0)

        with pytest.raises(ValueError):
            circuit_framework(-1)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_num_qubits_type_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test to see if the error is raised when the number of qubits
        is not an integer.
        """
        # Ensure the error is raised when the number of qubits is not an integer
        with pytest.raises(TypeError):
            circuit_framework(1.0) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_single_qubit_gate_from_range(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the single qubit gate when indices are passed as a range instance.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(3)

        # Define the qubit indices as a range of ints
        qubit_indices = range(3)

        # Apply the Pauli-X gate
        circuit.X(qubit_indices)

        # Define the checker
        checker_circuit = circuit_framework(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_single_qubit_gate_from_tuple(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the single qubit gate when indices are passed as a tuple instance.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(3)

        # Define the qubit indices as a tuple of ints
        qubit_indices = (0, 1, 2)

        # Apply the Pauli-X gate
        circuit.X(qubit_indices)

        # Define the checker
        checker_circuit = circuit_framework(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_single_qubit_gate_from_ndarray(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the single qubit gate when indices are passed as a numpy.ndarray instance.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(3)

        # Define the qubit indices as a ndarray of ints
        qubit_indices = np.array([0, 1, 2])

        # Apply the Pauli-X gate
        circuit.X(qubit_indices) # type: ignore

        # Define the checker
        checker_circuit = circuit_framework(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_qubit_type_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the qubit type error.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the Pauli-X gate
        with pytest.raises(TypeError):
            circuit.X("qubit") # type: ignore

        with pytest.raises(TypeError):
            circuit.X(["qubit1", "qubit2"]) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_duplicate_qubits(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the duplicate qubit error.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(3)

        # Apply the Pauli-X gate
        with pytest.raises(ValueError):
            circuit.X([0, 0])

        # Apply the MCX gate
        with pytest.raises(ValueError):
            circuit.MCX([0, 1], 1)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_qubit_out_of_range(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the qubit out of range error.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the Pauli-X gate
        with pytest.raises(IndexError):
            circuit.X(2)

        with pytest.raises(IndexError):
            circuit.X([0, 1])

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_control_out_of_range(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the control qubit out of range error.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply the CX gate
        with pytest.raises(IndexError):
            circuit.CX(2, 0)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_target_out_of_range(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the target qubit out of range error.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply the CX gate
        with pytest.raises(IndexError):
            circuit.CX(0, 2)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_angle_type_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the angle type error.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the RX gate
        with pytest.raises(TypeError):
            circuit.RX("angle", 0) # type: ignore

        with pytest.raises(TypeError):
            circuit.U3(["theta", "phi", "lambda"], 0) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("num_qubits", [
        1, 2, 3, 4, 5, 6, 7, 8
    ])
    def test_initialize(
            self,
            circuit_framework: Type[Circuit],
            num_qubits: int
        ) -> None:
        """ Test the state initialization.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        `num_qubits`: int
            The number of qubits in the circuit.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(num_qubits)

        # Define the statevector
        statevector = generate_random_state(num_qubits)

        # Initialize the circuit
        circuit.initialize(statevector, range(num_qubits))

        # Get the statevector of the circuit, and ensure it is correct
        assert_almost_equal(circuit.get_statevector(), statevector, 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("num_qubits", [
        1, 2, 3, 4, 5, 6
    ])
    def test_unitary(
            self,
            circuit_framework: Type[Circuit],
            num_qubits: int
        ) -> None:
        """ Test the unitary preparation gate.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        `num_qubits`: int
            The number of qubits in the circuit.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(num_qubits)

        # Apply the gate
        unitary = generate_random_unitary(num_qubits)
        circuit.unitary(unitary, range(num_qubits))

        # Define the unitary
        unitary = circuit.get_unitary()

        assert_almost_equal(circuit.get_unitary(), unitary, 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_get_global_phase(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the global phase extraction.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)
        circuit.GlobalPhase(1.4)

        # Get the global phase of the circuit, and ensure it is correct
        assert circuit.get_global_phase() == np.exp(3.2j)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_merge_global_phases(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the global phase merging.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)
        circuit.GlobalPhase(1.4)

        # Merge the global phases
        circuit.merge_global_phases()

        # Get the global phase of the circuit, and ensure it is correct
        assert circuit.get_global_phase() == np.exp(3.2j)
        assert repr(circuit) == circuit_framework.__name__ + "(num_qubits=1, circuit_log=[{'gate': 'GlobalPhase', 'angle': 3.2}])"

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_vertical_reverse(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the vertical reversal of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply GHZ state
        circuit.H(0)
        circuit.CX(0, 1)

        # Apply the vertical reverse operation
        circuit.vertical_reverse()

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        updated_circuit = circuit_framework(2)
        updated_circuit.H(1)
        updated_circuit.CX(1, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_horizontal_reverse(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the horizontal reversal of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply gates
        circuit.RX(np.pi, 0)
        circuit.CRX(np.pi, 0, 1)
        circuit.MCRX(np.pi, [0, 1], [2, 3])
        circuit.RY(np.pi, 0)
        circuit.CRY(np.pi, 0, 1)
        circuit.MCRY(np.pi, [0, 1], [2, 3])
        circuit.RZ(np.pi, 0)
        circuit.CRZ(np.pi, 0, 1)
        circuit.MCRZ(np.pi, [0, 1], [2, 3])
        circuit.S(0)
        circuit.T(0)
        circuit.CS(0, 1)
        circuit.CT(0, 1)
        circuit.MCS([0, 1], [2, 3])
        circuit.MCT([0, 1], [2, 3])
        circuit.U3([np.pi/2, np.pi/3, np.pi/4], 0)
        circuit.CU3([np.pi/2, np.pi/3, np.pi/4], 0, 1)
        circuit.MCU3([np.pi/2, np.pi/3, np.pi/4], [0, 1], [2, 3])
        circuit.CX(0, 1)
        circuit.MCX([0, 1], [2, 3])
        circuit.XPow(0.2, 0.1, 0)
        circuit.CXPow(0.2, 0.1, 0, 1)
        circuit.MCXPow(0.2, 0.1, [0, 1], [2, 3])
        circuit.YPow(0.2, 0.1, 0)
        circuit.CYPow(0.2, 0.1, 0, 1)
        circuit.MCYPow(0.2, 0.1, [0, 1], [2, 3])
        circuit.ZPow(0.2, 0.1, 0)
        circuit.CZPow(0.2, 0.1, 0, 1)
        circuit.MCZPow(0.2, 0.1, [0, 1], [2, 3])
        circuit.RXX(0.1, 0, 1)
        circuit.CRXX(0.1, 0, 1, 2)
        circuit.MCRXX(0.1, [0, 1], 2, 3)
        circuit.RYY(0.1, 0, 1)
        circuit.CRYY(0.1, 0, 1, 2)
        circuit.MCRYY(0.1, [0, 1], 2, 3)
        circuit.RZZ(0.1, 0, 1)
        circuit.CRZZ(0.1, 0, 1, 2)
        circuit.MCRZZ(0.1, [0, 1], 2, 3)

        # Apply the horizontal reverse operation
        circuit.horizontal_reverse()

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        updated_circuit = circuit_framework(4)
        updated_circuit.MCRZZ(-0.1, [0, 1], 2, 3)
        updated_circuit.CRZZ(-0.1, 0, 1, 2)
        updated_circuit.RZZ(-0.1, 0, 1)
        updated_circuit.MCRYY(-0.1, [0, 1], 2, 3)
        updated_circuit.CRYY(-0.1, 0, 1, 2)
        updated_circuit.RYY(-0.1, 0, 1)
        updated_circuit.MCRXX(-0.1, [0, 1], 2, 3)
        updated_circuit.CRXX(-0.1, 0, 1, 2)
        updated_circuit.RXX(-0.1, 0, 1)
        updated_circuit.MCZPow(-0.2, 0.1, [0, 1], [2, 3])
        updated_circuit.CZPow(-0.2, 0.1, 0, 1)
        updated_circuit.ZPow(-0.2, 0.1, 0)
        updated_circuit.MCYPow(-0.2, 0.1, [0, 1], [2, 3])
        updated_circuit.CYPow(-0.2, 0.1, 0, 1)
        updated_circuit.YPow(-0.2, 0.1, 0)
        updated_circuit.MCXPow(-0.2, 0.1, [0, 1], [2, 3])
        updated_circuit.CXPow(-0.2, 0.1, 0, 1)
        updated_circuit.XPow(-0.2, 0.1, 0)
        updated_circuit.MCX([0, 1], [2, 3])
        updated_circuit.CX(0, 1)
        updated_circuit.MCU3([-np.pi/2, -np.pi/4, -np.pi/3], [0, 1], [2, 3])
        updated_circuit.CU3([-np.pi/2, -np.pi/4, -np.pi/3], 0, 1)
        updated_circuit.U3([-np.pi/2, -np.pi/4, -np.pi/3], 0)
        updated_circuit.MCTdg([0, 1], [2, 3])
        updated_circuit.MCSdg([0, 1], [2, 3])
        updated_circuit.CTdg(0, 1)
        updated_circuit.CSdg(0, 1)
        updated_circuit.Tdg(0)
        updated_circuit.Sdg(0)
        updated_circuit.MCRZ(-np.pi, [0, 1], [2, 3])
        updated_circuit.CRZ(-np.pi, 0, 1)
        updated_circuit.RZ(-np.pi, 0)
        updated_circuit.MCRY(-np.pi, [0, 1], [2, 3])
        updated_circuit.CRY(-np.pi, 0, 1)
        updated_circuit.RY(-np.pi, 0)
        updated_circuit.MCRX(-np.pi, [0, 1], [2, 3])
        updated_circuit.CRX(-np.pi, 0, 1)
        updated_circuit.RX(-np.pi, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_horizontal_reverse_definition(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the horizontal reversal of the circuit definition.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define a custom gate without parameters such as angle, power, or diagonal
        # The implementation must be defined within the `with self.decompose_last(gate):` block
        # to provide `definition` key for the gate in the circuit log
        def custom_gate(self, qubit_indices: int | list[int]) -> None:
            gate = self.process_gate_params(gate="custom_gate", params=locals())

            with self.decompose_last(gate):
                self.RX(0.1, qubit_indices)

        # Apply the custom gate
        Circuit.custom_gate = custom_gate # type: ignore

        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply the custom gate
        circuit.custom_gate([0, 1]) # type: ignore

        # Apply the horizontal reverse operation
        circuit.horizontal_reverse()

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        checker_circuit = circuit_framework(2)
        checker_circuit.RX(-0.1, [0, 1])

        assert_almost_equal(circuit.get_unitary(), checker_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_add(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the addition of circuits.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instances
        circuit1 = circuit_framework(2)
        circuit2 = circuit_framework(2)

        # Apply the gates
        circuit1.CX(0, 1)
        circuit2.CY(0, 1)
        circuit2.H(0)

        # Add the two circuits
        circuit1.add(circuit2, [1, 0])

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        added_circuit = circuit_framework(2)
        added_circuit.CX(0, 1)
        added_circuit.CY(1, 0)
        added_circuit.H(1)

        assert circuit1 == added_circuit
        assert_almost_equal(circuit1.get_unitary(), added_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_add_fail(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the addition of circuits failure.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instances
        circuit1 = circuit_framework(2)
        circuit2 = circuit_framework(3)
        circuit3 = "circuit"

        # Ensure the error is raised when the type of the circuit is not correct
        with pytest.raises(TypeError):
            circuit1.add(circuit3, [0, 1]) # type: ignore

        # Ensure the error is raised when the number of qubits is not equal
        with pytest.raises(ValueError):
            circuit1.add(circuit2, [0, 1])

        # Ensure the error is raised when the qubit indices are not integers
        with pytest.raises(TypeError):
            circuit1.add(circuit2, [0, "index", 0.2]) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_transpile(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the transpilation of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        transpiled_circuit = circuit_framework(4)
        transpiled_circuit.MCX([0, 1], [2, 3])
        transpiled_circuit.transpile()

        assert_almost_equal(circuit.get_unitary(), transpiled_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_get_depth(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the depth of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Get the depth of the circuit, and ensure it is correct
        depth = circuit.get_depth()

        assert depth == 25

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_get_width(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the width of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Get the width of the circuit, and ensure it is correct
        width = circuit.get_width()

        assert width == 4

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_get_instructions(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the instructions of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Get the instructions of the circuit, and ensure they are correct
        instructions = circuit.get_instructions()

        instructions[0].pop("definition")

        assert instructions == [
            {"gate": "MCX", "control_indices": [0, 1], "target_indices": [2, 3]}
        ]

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_get_instructions_with_measurements(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the instructions of the circuit with measurements.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Apply the measurements
        circuit.measure([0, 1])

        # Get the instructions of the circuit, and ensure they are correct
        instructions = circuit.get_instructions()

        instructions[0].pop("definition")
        instructions[1].pop("definition")

        assert instructions == [
            {"gate": "MCX", "control_indices": [0, 1], "target_indices": [2, 3]},
            {"gate": "measure", "qubit_indices": [0, 1]}
        ]

        instructions = circuit.get_instructions(include_measurements=False)

        assert instructions == [
            {"gate": "MCX", "control_indices": [0, 1], "target_indices": [2, 3]}
        ]

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_compress(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the compression of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the RX gate
        circuit.RX(np.pi/2, 0)

        # Apply the U3 gate
        circuit.U3([np.pi/2, np.pi/2, np.pi/2], 0)

        # Compress the circuit
        circuit.compress(1.0)

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        compressed_circuit = circuit_framework(1)

        assert circuit == compressed_circuit
        assert_almost_equal(circuit.get_unitary(), compressed_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_compress_fail(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the compression of the circuit failure.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Ensure the error is raised when the compression factor is less than 0
        with pytest.raises(ValueError):
            circuit.compress(-1.0)

        # Ensure the error is raised when the compression factor is greater than 1
        with pytest.raises(ValueError):
            circuit.compress(2.0)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_change_mapping(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the mapping change of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Change the mapping of the circuit
        circuit.change_mapping([3, 2, 1, 0])

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        mapped_circuit = circuit_framework(4)
        mapped_circuit.MCX([3, 2], [1, 0])

        assert circuit == mapped_circuit
        assert_almost_equal(circuit.get_unitary(), mapped_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_change_mapping_indices_type_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the mapping change of the circuit failure.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Ensure the error is raised when the mapping indices are not integers
        with pytest.raises(TypeError):
            circuit.change_mapping([0, 1, "index", 2]) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_change_mapping_indices_value_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the mapping change of the circuit failure.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Ensure the error is raised when the mapping indices are out of range
        with pytest.raises(ValueError):
            circuit.change_mapping([0, 1, 2])

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_from_circuit(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `circuit.convert()` method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        circuit = circuit_framework(num_qubits=5)

        # Apply single qubit gates with both single index and multiple indices variations
        circuit.X(0)
        circuit.X([0, 1])
        circuit.Y(0)
        circuit.Y([0, 1])
        circuit.Z(0)
        circuit.Z([0, 1])
        circuit.H(0)
        circuit.H([0, 1])
        circuit.S(0)
        circuit.S([0, 1])
        circuit.Sdg(0)
        circuit.Sdg([0, 1])
        circuit.T(0)
        circuit.T([0, 1])
        circuit.Tdg(0)
        circuit.Tdg([0, 1])
        circuit.RX(0.5, 0)
        circuit.RX(0.5, [0, 1])
        circuit.RY(0.5, 0)
        circuit.RY(0.5, [0, 1])
        circuit.RZ(0.5, 0)
        circuit.RZ(0.5, [0, 1])
        circuit.Phase(0.5, 0)
        circuit.Phase(0.5, [0, 1])
        circuit.U3([0.1, 0.2, 0.3], 0)
        circuit.SWAP(0, 1)

        # Apply controlled gates
        circuit.CX(0, 1)
        circuit.CY(0, 1)
        circuit.CZ(0, 1)
        circuit.CH(0, 1)
        circuit.CS(0, 1)
        circuit.CSdg(0, 1)
        circuit.CT(0, 1)
        circuit.CTdg(0, 1)
        circuit.CRX(0.5, 0, 1)
        circuit.CRY(0.5, 0, 1)
        circuit.CRZ(0.5, 0, 1)
        circuit.CPhase(0.5, 0, 1)
        circuit.CU3([0.1, 0.2, 0.3], 0, 1)
        circuit.CSWAP(0, 1, 2)

        # Apply multi-controlled gates with both single index and multiple indices variations
        circuit.MCX(0, 1)
        circuit.MCX([0, 1], 2)
        circuit.MCX(0, [1, 2])
        circuit.MCX([0, 1], [2, 3])

        circuit.MCY(0, 1)
        circuit.MCY([0, 1], 2)
        circuit.MCY(0, [1, 2])
        circuit.MCY([0, 1], [2, 3])

        circuit.MCZ(0, 1)
        circuit.MCZ([0, 1], 2)
        circuit.MCZ(0, [1, 2])
        circuit.MCZ([0, 1], [2, 3])

        circuit.MCH(0, 1)
        circuit.MCH([0, 1], 2)
        circuit.MCH(0, [1, 2])
        circuit.MCH([0, 1], [2, 3])

        circuit.MCS(0, 1)
        circuit.MCS([0, 1], 2)
        circuit.MCS(0, [1, 2])
        circuit.MCS([0, 1], [2, 3])

        circuit.MCSdg(0, 1)
        circuit.MCSdg([0, 1], 2)
        circuit.MCSdg(0, [1, 2])
        circuit.MCSdg([0, 1], [2, 3])

        circuit.MCT(0, 1)
        circuit.MCT([0, 1], 2)
        circuit.MCT(0, [1, 2])
        circuit.MCT([0, 1], [2, 3])

        circuit.MCTdg(0, 1)
        circuit.MCTdg([0, 1], 2)
        circuit.MCTdg(0, [1, 2])
        circuit.MCTdg([0, 1], [2, 3])

        circuit.MCRX(0.5, 0, 1)
        circuit.MCRX(0.5, [0, 1], 2)
        circuit.MCRX(0.5, 0, [1, 2])
        circuit.MCRX(0.5, [0, 1], [2, 3])

        circuit.MCRY(0.5, 0, 1)
        circuit.MCRY(0.5, [0, 1], 2)
        circuit.MCRY(0.5, 0, [1, 2])
        circuit.MCRY(0.5, [0, 1], [2, 3])

        circuit.MCRZ(0.5, 0, 1)
        circuit.MCRZ(0.5, [0, 1], 2)
        circuit.MCRZ(0.5, 0, [1, 2])
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        circuit.MCPhase(0.5, 0, 1)
        circuit.MCPhase(0.5, [0, 1], 2)
        circuit.MCPhase(0.5, 0, [1, 2])
        circuit.MCPhase(0.5, [0, 1], [2, 3])

        circuit.MCU3([0.1, 0.2, 0.3], 0, 1)
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], 2)
        circuit.MCU3([0.1, 0.2, 0.3], 0, [1, 2])
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        circuit.MCSWAP(0, 1, 2)
        circuit.MCSWAP([0, 1], 2, 3)

        # Apply global phase
        circuit.GlobalPhase(0.5)

        # Apply measurement
        circuit.measure(0)
        circuit.measure([1, 2])

        # Convert the circuit
        converted_circuits = [
            circuit.convert(circuit_framework) for circuit_framework in CIRCUIT_FRAMEWORKS
        ]

        # Check the converted circuit
        for converted_circuit in converted_circuits:
            assert circuit == converted_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_convert_type_error(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `circuit.convert()` method failure.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        circuit = circuit_framework(num_qubits=5)

        # Ensure the error is raised when the type of the circuit is not correct
        with pytest.raises(TypeError):
            circuit.convert("circuit") # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_reset(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the reset of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply the Hadamard gate
        circuit.H(0)

        # Reset the circuit
        circuit.reset()

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        reset_circuit = circuit_framework(2)

        assert circuit == reset_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_remove_measurement(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the removal of measurement gate.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)
        no_measurement_circuit = circuit_framework(2)

        # Apply the measurement gate
        circuit.measure([0, 1])

        # Ensure both qubits are measured
        assert circuit.measured_qubits == {0, 1}

        # Remove the measurement gate
        updated_circuit = circuit.remove_measurements(inplace=False)

        # Ensure no qubits are measured
        assert len(updated_circuit.measured_qubits) == 0

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        assert updated_circuit == no_measurement_circuit

        circuit.remove_measurements(inplace=True)

        # Ensure no qubits are measured
        assert len(circuit.measured_qubits) == 0

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        assert circuit == no_measurement_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_getitem(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `__getitem__` dunder method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply a series of gates
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.RX(np.pi/2, 0)
        circuit.RY(np.pi/2, 1)

        # Test the `__getitem__` dunder method
        new_circuit = circuit_framework(2)
        new_circuit = circuit[1:]

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        checker_circuit = circuit_framework(2)
        checker_circuit.CX(0, 1)
        checker_circuit.RX(np.pi/2, 0)
        checker_circuit.RY(np.pi/2, 1)

        assert checker_circuit == new_circuit

        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply a series of gates
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.RX(np.pi/2, 0)

        # Test the `__getitem__` dunder method
        new_circuit = circuit_framework(2)
        new_circuit = circuit[1]

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        checker_circuit = circuit_framework(2)
        checker_circuit.CX(0, 1)

        assert checker_circuit == new_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_setitem(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `__setitem__` dunder method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply a series of gates
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.RX(np.pi/2, 0)

        # Test the `__setitem__` dunder method
        new_circuit = circuit_framework(2)
        new_circuit.H(0)
        new_circuit.CX(0, 1)

        new_circuit[1] = circuit[1]

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        checker_circuit = circuit_framework(2)
        checker_circuit.H(0)
        checker_circuit.CX(0, 1)

        assert checker_circuit == new_circuit

        # Define the `quick.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply a series of gates
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.RX(np.pi/2, 0)
        circuit.RY(np.pi/2, 1)

        # Test the `__setitem__` dunder method
        new_circuit = circuit_framework(2)
        new_circuit[:] = circuit[-1]

        # Define the equivalent `quick.circuit.Circuit` instance, and
        # ensure they are equivalent
        checker_circuit = circuit_framework(2)
        checker_circuit.RY(np.pi/2, 1)

        assert checker_circuit == new_circuit

    @pytest.mark.parametrize("circuit_frameworks", [CIRCUIT_FRAMEWORKS])
    def test_eq(
            self,
            circuit_frameworks: list[Type[Circuit]]
        ) -> None:
        """ Test the `__eq__` dunder method.

        Parameters
        ----------
        `circuit_frameworks`: list[type[quick.circuit.Circuit]]
            The circuit frameworks to test.
        """
        circuits = [circuit_framework(2) for circuit_framework in circuit_frameworks]

        # Define the Bell state
        for circuit in circuits:
            circuit.H(0)
            circuit.CX(0, 1)

        # Test the equality of the circuits
        for circuit_1, circuit_2 in zip(circuits[0:-1:], circuits[1::]):
            assert circuit_1 == circuit_2

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_len(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `__len__` dunder method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the circuits
        circuit = circuit_framework(2)

        # Define the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Test the length of the circuit
        assert len(circuit) == 2

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_str(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `__str__` dunder method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the circuits
        circuit = circuit_framework(2)

        # Define the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Test the string representation of the circuits
        assert str(circuit) == f"{circuit_framework.__name__}(num_qubits=2)"

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_generate_calls(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `generate_calls` method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the circuits
        circuit = circuit_framework(2)

        # Define the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Test the generated calls
        assert circuit.generate_calls() == 'circuit.H(qubit_indices=0)\ncircuit.CX(control_index=0, target_index=1)\n'

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_repr(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `__repr__` dunder method.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the circuits
        circuit = circuit_framework(2)

        # Define the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Test the string representation of the circuits
        circuit_checker = (
            f"{circuit_framework.__name__}(num_qubits=2, "
            "circuit_log=[{'gate': 'H', 'qubit_indices': 0}, "
            "{'gate': 'CX', 'control_index': 0, 'target_index': 1}])"
        )
        assert repr(circuit) == circuit_checker

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_custom_gate(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the custom gate functionality.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the custom gate
        def custom_gate(self, qubit_indices: int | list[int]) -> None:
            gate = self.process_gate_params(gate="custom_gate", params=locals())

            with self.decompose_last(gate):
                self.RX(0.1, qubit_indices)

        # Add the custom gate
        Circuit.custom_gate = custom_gate # type: ignore

        # Define the circuits
        circuit = circuit_framework(2)

        # Apply the custom gate
        circuit.custom_gate([0, 1]) # type: ignore

        # Define checker circuit
        checker_circuit = circuit_framework(2)
        checker_circuit.RX(0.1, [0, 1])

        # Test the generated calls
        assert circuit.generate_calls() == 'circuit.custom_gate(qubit_indices=[0, 1])\n'
        assert_almost_equal(circuit.get_unitary(), checker_circuit.get_unitary())

        reverse_circuit = circuit_framework(2)
        reverse_circuit.RX(-0.1, [0, 1])

        # Test the horizontal reverse operation
        circuit = circuit_framework(2)
        circuit.custom_gate([0, 1]) # type: ignore
        circuit.horizontal_reverse()

        assert_almost_equal(circuit.get_unitary(), reverse_circuit.get_unitary())