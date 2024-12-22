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

__all__ = ["TestCircuitBase"]

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from scipy.stats import unitary_group
from typing import Type

from qickit.circuit import Circuit

from tests.circuit import CIRCUIT_FRAMEWORKS
from tests.circuit.utils import generate_random_state


class TestCircuitBase:
    """ `tests.circuit.TestCircuitBase` class is a tester for the base functionality
    of the `qickit.circuit.Circuit` class.
    """
    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_remove_measurement(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the removal of measurement gate.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(2)
        no_measurement_circuit = circuit_framework(2)

        # Apply the measurement gate
        circuit.measure([0, 1])

        # Ensure both qubits are measured
        assert circuit.measured_qubits == {0, 1}

        # Remove the measurement gate
        circuit = circuit._remove_measurements()

        # Ensure no qubits are measured
        assert len(circuit.measured_qubits) == 0

        # Define the equivalent `qickit.circuit.Circuit` instance, and
        # ensure they are equivalent
        assert circuit == no_measurement_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_qubit_out_of_range(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the qubit out of range error.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the Pauli-X gate
        with pytest.raises(IndexError):
            circuit.X(2)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_control_out_of_range(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the control qubit out of range error.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply the CX gate
        with pytest.raises(IndexError):
            circuit.CX(0, 2)

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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        `num_qubits`: int
            The number of qubits in the circuit.
        """
        # Define the `qickit.circuit.Circuit` instance
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        `num_qubits`: int
            The number of qubits in the circuit.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(num_qubits)

        # Apply the gate
        unitary = unitary_group.rvs(2 ** num_qubits).astype(complex)
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)
        circuit.GlobalPhase(1.4)

        # Get the global phase of the circuit, and ensure it is correct
        assert circuit.get_global_phase() == np.exp(3.2j)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_vertical_reverse(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the vertical reversal of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply GHZ state
        circuit.H(0)
        circuit.CX(0, 1)

        # Apply the vertical reverse operation
        circuit.vertical_reverse()

        # Define the equivalent `qickit.circuit.Circuit` instance, and
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
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

        # Define the equivalent `qickit.circuit.Circuit` instance, and
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
    def test_add(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the addition of circuits.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instances
        circuit1 = circuit_framework(2)
        circuit2 = circuit_framework(2)

        # Apply the gates
        circuit1.CX(0, 1)
        circuit2.CY(0, 1)
        circuit2.H(0)

        # Add the two circuits
        circuit1.add(circuit2, [1, 0])

        # Define the equivalent `qickit.circuit.Circuit` instance, and
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instances
        circuit1 = circuit_framework(2)
        circuit2 = circuit_framework(3)
        circuit3 = "circuit"

        # Ensure the error is raised when the type of the circuit is not correct
        with pytest.raises(TypeError):
            circuit1.add(circuit3, [0, 1]) # type: ignore

        # Ensure the error is raised when the number of qubits is not equal
        with pytest.raises(ValueError):
            circuit1.add(circuit2, [0, 1])

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_transpile(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the transpilation of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `qickit.circuit.Circuit` instance, and
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Get the width of the circuit, and ensure it is correct
        width = circuit.get_width()

        assert width == 4

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_compress(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the compression of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(1)

        # Apply the RX gate
        circuit.RX(np.pi/2, 0)

        # Compress the circuit
        circuit.compress(1.0)

        # Define the equivalent `qickit.circuit.Circuit` instance, and
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
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
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Change the mapping of the circuit
        circuit.change_mapping([3, 2, 1, 0])

        # Define the equivalent `qickit.circuit.Circuit` instance, and
        # ensure they are equivalent
        mapped_circuit = circuit_framework(4)
        mapped_circuit.MCX([3, 2], [1, 0])

        assert circuit == mapped_circuit
        assert_almost_equal(circuit.get_unitary(), mapped_circuit.get_unitary(), 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_from_circuit(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `circuit.convert()` method.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
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
    def test_reset(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the reset of the circuit.

        Parameters
        ----------
        `circuit_framework`: type[qickit.circuit.Circuit]
            The circuit framework to test.
        """
        # Define the `qickit.circuit.Circuit` instance
        circuit = circuit_framework(2)

        # Apply the Hadamard gate
        circuit.H(0)

        # Reset the circuit
        circuit.reset()

        # Define the equivalent `qickit.circuit.Circuit` instance, and
        # ensure they are equivalent
        reset_circuit = circuit_framework(2)

        assert circuit == reset_circuit

    @pytest.mark.parametrize("circuit_frameworks", [CIRCUIT_FRAMEWORKS])
    def test_eq(
            self,
            circuit_frameworks: list[Type[Circuit]]
        ) -> None:
        """ Test the `__eq__` dunder method.
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
        """
        # Define the circuits
        circuit = circuit_framework(2)

        # Define the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Test the string representation of the circuits
        assert str(circuit) == f"{circuit_framework.__name__}(num_qubits=2)"

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_repr(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `__repr__` dunder method.
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