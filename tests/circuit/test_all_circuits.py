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

__all__ = ['TestAllCircuits']

from numpy.testing import assert_almost_equal

# QICKIT imports
from qickit.circuit import (CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit)
from tests.circuit import Template


class TestAllCircuits(Template):
    """ `tests.circuit.TestAllCircuits` is the tester class for ensuring all frameworks return the same result.
    """
    def test_X(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-X gate
        cirq_circuit.X(0)
        pennylane_circuit.X(0)
        qiskit_circuit.X(0)
        tket_circuit.X(0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_Y(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-Y gate
        cirq_circuit.Y(0)
        pennylane_circuit.Y(0)
        qiskit_circuit.Y(0)
        tket_circuit.Y(0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_Z(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-Z gate
        cirq_circuit.Z(0)
        pennylane_circuit.Z(0)
        qiskit_circuit.Z(0)
        tket_circuit.Z(0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_H(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Hadamard gate
        cirq_circuit.H(0)
        pennylane_circuit.H(0)
        qiskit_circuit.H(0)
        tket_circuit.H(0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_S(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the S gate
        cirq_circuit.S(0)
        pennylane_circuit.S(0)
        qiskit_circuit.S(0)
        tket_circuit.S(0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_T(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the T gate
        cirq_circuit.T(0)
        pennylane_circuit.T(0)
        qiskit_circuit.T(0)
        tket_circuit.T(0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_RX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-X gate
        cirq_circuit.RX(0.5, 0)
        pennylane_circuit.RX(0.5, 0)
        qiskit_circuit.RX(0.5, 0)
        tket_circuit.RX(0.5, 0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_RY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the RY gate
        cirq_circuit.RY(0.5, 0)
        pennylane_circuit.RY(0.5, 0)
        qiskit_circuit.RY(0.5, 0)
        tket_circuit.RY(0.5, 0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_RZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the RZ gate
        cirq_circuit.RZ(0.5, 0)
        pennylane_circuit.RZ(0.5, 0)
        qiskit_circuit.RZ(0.5, 0)
        tket_circuit.RZ(0.5, 0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_U3(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the U3 gate
        cirq_circuit.U3([0.1, 0.2, 0.3], 0)
        pennylane_circuit.U3([0.1, 0.2, 0.3], 0)
        qiskit_circuit.U3([0.1, 0.2, 0.3], 0)
        tket_circuit.U3([0.1, 0.2, 0.3], 0)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_SWAP(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the SWAP gate
        cirq_circuit.SWAP(0, 1)
        pennylane_circuit.SWAP(0, 1)
        qiskit_circuit.SWAP(0, 1)
        tket_circuit.SWAP(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CX gate
        cirq_circuit.CX(0, 1)
        pennylane_circuit.CX(0, 1)
        qiskit_circuit.CX(0, 1)
        tket_circuit.CX(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CY gate
        cirq_circuit.CY(0, 1)
        pennylane_circuit.CY(0, 1)
        qiskit_circuit.CY(0, 1)
        tket_circuit.CY(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CZ gate
        cirq_circuit.CZ(0, 1)
        pennylane_circuit.CZ(0, 1)
        qiskit_circuit.CZ(0, 1)
        tket_circuit.CZ(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CH(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CH gate
        cirq_circuit.CH(0, 1)
        pennylane_circuit.CH(0, 1)
        qiskit_circuit.CH(0, 1)
        tket_circuit.CH(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CS(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CS gate
        cirq_circuit.CS(0, 1)
        pennylane_circuit.CS(0, 1)
        qiskit_circuit.CS(0, 1)
        tket_circuit.CS(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CT(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CT gate
        cirq_circuit.CT(0, 1)
        pennylane_circuit.CT(0, 1)
        qiskit_circuit.CT(0, 1)
        tket_circuit.CT(0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CRX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CRX gate
        cirq_circuit.CRX(0.5, 0, 1)
        pennylane_circuit.CRX(0.5, 0, 1)
        qiskit_circuit.CRX(0.5, 0, 1)
        tket_circuit.CRX(0.5, 0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CRY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CRY gate
        cirq_circuit.CRY(0.5, 0, 1)
        pennylane_circuit.CRY(0.5, 0, 1)
        qiskit_circuit.CRY(0.5, 0, 1)
        tket_circuit.CRY(0.5, 0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CRZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CRZ gate
        cirq_circuit.CRZ(0.5, 0, 1)
        pennylane_circuit.CRZ(0.5, 0, 1)
        qiskit_circuit.CRZ(0.5, 0, 1)
        tket_circuit.CRZ(0.5, 0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CU3(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CU3 gate
        cirq_circuit.CU3([0.1, 0.2, 0.3], 0, 1)
        pennylane_circuit.CU3([0.1, 0.2, 0.3], 0, 1)
        qiskit_circuit.CU3([0.1, 0.2, 0.3], 0, 1)
        tket_circuit.CU3([0.1, 0.2, 0.3], 0, 1)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_CSWAP(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(3, 3)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(3, 3)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(3, 3)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(3, 3)

        # Apply the CSWAP gate
        cirq_circuit.CSWAP(0, 1, 2)
        pennylane_circuit.CSWAP(0, 1, 2)
        qiskit_circuit.CSWAP(0, 1, 2)
        tket_circuit.CSWAP(0, 1, 2)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCX gate
        cirq_circuit.MCX([0, 1], [2, 3])
        pennylane_circuit.MCX([0, 1], [2, 3])
        qiskit_circuit.MCX([0, 1], [2, 3])
        tket_circuit.MCX([0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCY gate
        cirq_circuit.MCY([0, 1], [2, 3])
        pennylane_circuit.MCY([0, 1], [2, 3])
        qiskit_circuit.MCY([0, 1], [2, 3])
        tket_circuit.MCY([0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCZ gate
        cirq_circuit.MCZ([0, 1], [2, 3])
        pennylane_circuit.MCZ([0, 1], [2, 3])
        qiskit_circuit.MCZ([0, 1], [2, 3])
        tket_circuit.MCZ([0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCH(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCH gate
        cirq_circuit.MCH([0, 1], [2, 3])
        pennylane_circuit.MCH([0, 1], [2, 3])
        qiskit_circuit.MCH([0, 1], [2, 3])
        tket_circuit.MCH([0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCS(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCS gate
        cirq_circuit.MCS([0, 1], [2, 3])
        pennylane_circuit.MCS([0, 1], [2, 3])
        qiskit_circuit.MCS([0, 1], [2, 3])
        tket_circuit.MCS([0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCT(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCT gate
        cirq_circuit.MCT([0, 1], [2, 3])
        pennylane_circuit.MCT([0, 1], [2, 3])
        qiskit_circuit.MCT([0, 1], [2, 3])
        tket_circuit.MCT([0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCRX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCRX gate
        cirq_circuit.MCRX(0.5, [0, 1], [2, 3])
        pennylane_circuit.MCRX(0.5, [0, 1], [2, 3])
        qiskit_circuit.MCRX(0.5, [0, 1], [2, 3])
        tket_circuit.MCRX(0.5, [0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCRY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCRY gate
        cirq_circuit.MCRY(0.5, [0, 1], [2, 3])
        pennylane_circuit.MCRY(0.5, [0, 1], [2, 3])
        qiskit_circuit.MCRY(0.5, [0, 1], [2, 3])
        tket_circuit.MCRY(0.5, [0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCRZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCRZ gate
        cirq_circuit.MCRZ(0.5, [0, 1], [2, 3])
        pennylane_circuit.MCRZ(0.5, [0, 1], [2, 3])
        qiskit_circuit.MCRZ(0.5, [0, 1], [2, 3])
        tket_circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCU3(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCU3 gate
        cirq_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])
        pennylane_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])
        qiskit_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])
        tket_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_MCSWAP(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCSWAP gate
        cirq_circuit.MCSWAP([0, 1], 2, 3)
        pennylane_circuit.MCSWAP([0, 1], 2, 3)
        qiskit_circuit.MCSWAP([0, 1], 2, 3)
        tket_circuit.MCSWAP([0, 1], 2, 3)

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_GlobalPhase(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the GlobalPhase gate
        cirq_circuit.GlobalPhase(1.8)
        pennylane_circuit.GlobalPhase(1.8)
        qiskit_circuit.GlobalPhase(1.8)
        tket_circuit.GlobalPhase(1.8)

        print("Cirq")
        print(cirq_circuit.get_unitary())
        print("Pennylane")
        print(pennylane_circuit.get_unitary())
        print("Qiskit")
        print(qiskit_circuit.get_unitary())
        print("TKET")
        print(tket_circuit.get_unitary())

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_measure(self) -> None:
        pass

    def test_unitary(self) -> None:
        pass

    def test_vertical_reverse(self) -> None:
        pass

    def test_horizontal_reverse(self) -> None:
        pass

    def test_add(self) -> None:
        pass

    def test_transpile(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.circuit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.circuit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.circuit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCX gate
        cirq_circuit.MCX([0, 1], [2, 3])
        pennylane_circuit.MCX([0, 1], [2, 3])
        qiskit_circuit.MCX([0, 1], [2, 3])
        tket_circuit.MCX([0, 1], [2, 3])

        # Transpile
        cirq_circuit.transpile()
        pennylane_circuit.transpile()
        qiskit_circuit.transpile()
        tket_circuit.transpile()

        # Ensure they are equivalent
        assert_almost_equal(cirq_circuit.get_unitary(), pennylane_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), qiskit_circuit.get_unitary(), 8)
        assert_almost_equal(cirq_circuit.get_unitary(), tket_circuit.get_unitary(), 8)

    def test_get_depth(self) -> None:
        pass

    def test_get_width(self) -> None:
        pass

    def test_compress(self) -> None:
        pass

    def test_change_mapping(self) -> None:
        pass