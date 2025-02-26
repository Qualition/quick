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

__all__ = ["TestFromQiskit"]

import numpy as np
from numpy.testing import assert_almost_equal
from qiskit import QuantumCircuit # type: ignore
from qiskit.circuit.library import U3Gate, CXGate, GlobalPhaseGate
from qiskit.quantum_info import Operator # type: ignore

from quick.circuit import Circuit, QiskitCircuit


class TestFromQiskit:
    """ `tests.circuit.TestFromQiskit` tests the `.from_qiskit` method.
    """
    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.append(U3Gate(0.1, 0.2, 0.3), [0])

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        assert_almost_equal(
            quick_circuit.get_unitary(),
            np.array(Operator(qiskit_circuit).data),
            8
        )

    def test_CX(self) -> None:
        """ Test the CX gate.
        """
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.append(CXGate(), [0, 1])

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        assert_almost_equal(
            quick_circuit.get_unitary(),
            np.array(Operator(qiskit_circuit).data),
            8
        )

    def test_global_phase(self) -> None:
        """ Test the global phase gate.
        """
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.append(GlobalPhaseGate(0.1), [])

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        assert_almost_equal(
            quick_circuit.get_unitary(),
            np.array(Operator(qiskit_circuit).data),
            8
        )

    def test_single_measurement(self) -> None:
        """ Test the single qubit measurement.
        """
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.measure(0, 0)

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent quick circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.measure(0)
        assert quick_circuit == check_circuit

    def test_multiple_measurement(self) -> None:
        """ Test the multi-qubit measurement.
        """
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.measure(0, 0)
        qiskit_circuit.measure(1, 1)

        # Convert the Qiskit circuit to a quick circuit
        quick_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent quick circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.measure(0)
        check_circuit.measure(1)
        assert quick_circuit == check_circuit