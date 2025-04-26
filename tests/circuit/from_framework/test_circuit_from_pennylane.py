# Copyright 2023-2025 Qualition Computing LLC.
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

__all__ = ["TestFromPennylane"]

import numpy as np
from numpy.testing import assert_almost_equal
import pennylane as qml # type: ignore

from quick.circuit import Circuit, PennylaneCircuit


class TestFromPennylane:
    """ `tests.circuit.TestFromPennylane` tests the `.from_pennylane` method.
    """
    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the Pennylane circuit
        def pennylane_circuit():
            qml.RX(phi=0.1, wires=0) # type: ignore

        unitary = np.array(
            qml.matrix(pennylane_circuit, wire_order=[0])(), dtype=complex # type: ignore
        )

        # Convert the Pennylane circuit to a quick circuit
        quick_circuit = Circuit.from_pennylane(
            qml.QNode(
                pennylane_circuit,
                device=qml.device("default.qubit", 1)
            ),
            PennylaneCircuit
        )

        assert_almost_equal(
            quick_circuit.get_unitary(),
            unitary,
            8
        )

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the Pennylane circuit
        def pennylane_circuit():
            qml.RZ(phi=0.1, wires=0) # type: ignore

        unitary = np.array(
            qml.matrix(pennylane_circuit, wire_order=0)(), dtype=complex # type: ignore
        )

        # Convert the Pennylane circuit to a quick circuit
        quick_circuit = Circuit.from_pennylane(
            qml.QNode(
                pennylane_circuit,
                device=qml.device("default.qubit", 1)
            ),
            PennylaneCircuit
        )

        assert_almost_equal(
            quick_circuit.get_unitary(),
            unitary,
            8
        )

    def test_CX(self) -> None:
        """ Test the CX gate.
        """
        # Define the Pennylane circuit
        def pennylane_circuit():
            qml.CNOT(wires=[0, 1]) # type: ignore

        unitary = np.array(
            qml.matrix(pennylane_circuit, wire_order=[1, 0])(), dtype=complex # type: ignore
        )

        # Convert the Pennylane circuit to a quick circuit
        quick_circuit = Circuit.from_pennylane(
            qml.QNode(
                pennylane_circuit,
                device=qml.device("default.qubit", 2)
            ),
            PennylaneCircuit
        )

        assert_almost_equal(
            quick_circuit.get_unitary(),
            unitary,
            8
        )

    def test_global_phase(self) -> None:
        """ Test the global phase gate.
        """
        # Define the Pennylane circuit
        def pennylane_circuit():
            qml.I(0) # type: ignore
            qml.GlobalPhase(0.1) # type: ignore

        unitary = np.array(
            qml.matrix(pennylane_circuit, wire_order=0)(), dtype=complex # type: ignore
        )

        # Convert the Pennylane circuit to a quick circuit
        quick_circuit = Circuit.from_pennylane(
            qml.QNode(
                pennylane_circuit,
                device=qml.device("default.qubit", 1)
            ),
            PennylaneCircuit
        )

        assert_almost_equal(
            quick_circuit.get_unitary(),
            unitary,
            8
        )

    def test_single_measurement(self) -> None:
        """ Test the single qubit measurement.
        """
        # Define the Pennylane circuit
        def pennylane_circuit():
            qml.measure(0)

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_pennylane(
            qml.QNode(
                pennylane_circuit,
                device=qml.device("default.qubit", 1)
            ),
            PennylaneCircuit
        )

        # Define the equivalent quick circuit, and ensure
        # that the two circuits are equal
        check_circuit = PennylaneCircuit(1)
        check_circuit.measure(0)
        assert quick_circuit == check_circuit

    def test_multiple_measurement(self) -> None:
        """ Test the multi-qubit measurement.
        """
        # Define the Pennylane circuit
        def pennylane_circuit():
            qml.measure(0)
            qml.measure(1)

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_pennylane(
            qml.QNode(
                pennylane_circuit,
                device=qml.device("default.qubit", 2)
            ),
            PennylaneCircuit
        )

        # Define the equivalent quick circuit, and ensure
        # that the two circuits are equal
        check_circuit = PennylaneCircuit(2)
        check_circuit.measure(0)
        check_circuit.measure(1)
        assert quick_circuit == check_circuit