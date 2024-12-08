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

__all__ = ["TestFromCirq"]

import cirq
from numpy.testing import assert_almost_equal

from qickit.circuit import Circuit, CirqCircuit


class TestFromCirq:
    """ `tests.circuit.TestFromCirq` tests the `.from_cirq` method.
    """
    def test_PhasedXZ(self) -> None:
        """ Test the PhasedXZ gate.
        """
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(cirq.PhasedXZGate(x_exponent=0.1, z_exponent=0.2, axis_phase_exponent=0.3)(qr[0]))

        # Convert the Cirq circuit to a Qickit circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, CirqCircuit)

        assert_almost_equal(
            qickit_circuit.get_unitary(),
            cirq_circuit.unitary(),
            8
        )

    def test_CZ(self) -> None:
        """ Test the CZ gate.
        """
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(cirq.CZ(qr[0], qr[1]))

        # Convert the Cirq circuit to a Qickit circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, CirqCircuit)

        qickit_circuit.vertical_reverse()

        assert_almost_equal(
            qickit_circuit.get_unitary(),
            cirq_circuit.unitary(),
            8
        )

    def test_single_measurement(self) -> None:
        """ Test the single qubit measurement.
        """
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(cirq.measure(qr[0]))

        # Convert the Cirq circuit to a Qickit circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, CirqCircuit)

        # Define the equivalent Qickit circuit, and ensure
        # that the two circuits are equal
        check_circuit = CirqCircuit(1)
        check_circuit.measure(0)
        assert qickit_circuit == check_circuit

    def test_multiple_measurement(self) -> None:
        """ Test the multi-qubit measurement.k
        """
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(cirq.measure(qr[0], qr[1]))

        # Convert the Cirq circuit to a Qickit circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, CirqCircuit)

        # Define the equivalent Qickit circuit, and ensure
        # that the two circuits are equal
        check_circuit = CirqCircuit(2)
        check_circuit.measure([0, 1])
        assert qickit_circuit == check_circuit