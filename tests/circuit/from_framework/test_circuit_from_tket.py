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

__all__ = ["TestFromTKET"]

from numpy.testing import assert_almost_equal
from pytket import Circuit as TKCircuit
from pytket import OpType

from qickit.circuit import Circuit, TKETCircuit


class TestFromTKET:
    def test_U3(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        angles = [0.1, 0.2, 0.3]
        tket_circuit.add_gate(OpType.U3, angles, [0])

        # Convert the TKET circuit to a Qickit circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        assert_almost_equal(
            qickit_circuit.get_unitary(),
            tket_circuit.get_unitary(),
            8
        )

    def test_CX(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CX, [0, 1])

        # Convert the TKET circuit to a Qickit circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        qickit_circuit.vertical_reverse()

        assert_almost_equal(
            qickit_circuit.get_unitary(),
            tket_circuit.get_unitary(),
            8
        )

    def test_GlobalPhase(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_phase(0.5)

        # Convert the TKET circuit to a Qickit circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        assert_almost_equal(
            qickit_circuit.get_unitary(),
            tket_circuit.get_unitary(),
            8
        )

    def test_single_measurement(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.Measure(0, 0)

        # Convert the TKET circuit to a Qickit circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent Qickit circuit, and ensure
        # that the two circuits are equal
        checker_circuit = TKETCircuit(1)
        checker_circuit.measure(0)
        assert qickit_circuit == checker_circuit

    def test_multiple_measurement(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.Measure(0, 0)
        tket_circuit.Measure(1, 1)

        # Convert the TKET circuit to a Qickit circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent Qickit circuit, and ensure
        # that the two circuits are equal
        checker_circuit = TKETCircuit(2)
        checker_circuit.measure(0)
        checker_circuit.measure(1)
        assert qickit_circuit == checker_circuit