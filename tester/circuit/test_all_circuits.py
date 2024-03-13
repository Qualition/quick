# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the GNU Version 3.0 (the "License");
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

import numpy as np

# QICKIT imports
from qickit.circuit import (CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit)
from tester.circuit import TestCircuit


class TestAllCircuits(TestCircuit):
    """ `qickit.TestAllCircuits` is the tester class for ensuring all frameworks return the same result.
    """
    def test_circuit_initialization(self) -> None:
        pass

    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-X gate
        cirq_circuit.X(0)
        pennylane_circuit.X(0)
        qiskit_circuit.X(0)
        tket_circuit.X(0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-Y gate
        cirq_circuit.Y(0)
        pennylane_circuit.Y(0)
        qiskit_circuit.Y(0)
        tket_circuit.Y(0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-Z gate
        cirq_circuit.Z(0)
        pennylane_circuit.Z(0)
        qiskit_circuit.Z(0)
        tket_circuit.Z(0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Hadamard gate
        cirq_circuit.H(0)
        pennylane_circuit.H(0)
        qiskit_circuit.H(0)
        tket_circuit.H(0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_S(self) -> None:
        """ Test the S gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the S gate
        cirq_circuit.S(0)
        pennylane_circuit.S(0)
        qiskit_circuit.S(0)
        tket_circuit.S(0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_T(self) -> None:
        """ Test the T gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the T gate
        cirq_circuit.T(0)
        pennylane_circuit.T(0)
        qiskit_circuit.T(0)
        tket_circuit.T(0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the Pauli-X gate
        cirq_circuit.RX(0.5, 0)
        pennylane_circuit.RX(0.5, 0)
        qiskit_circuit.RX(0.5, 0)
        tket_circuit.RX(0.5, 0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the RY gate
        cirq_circuit.RY(0.5, 0)
        pennylane_circuit.RY(0.5, 0)
        qiskit_circuit.RY(0.5, 0)
        tket_circuit.RY(0.5, 0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the RZ gate
        cirq_circuit.RZ(0.5, 0)
        pennylane_circuit.RZ(0.5, 0)
        qiskit_circuit.RZ(0.5, 0)
        tket_circuit.RZ(0.5, 0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(1, 1)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(1, 1)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(1, 1)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(1, 1)

        # Apply the U3 gate
        cirq_circuit.U3([0.5, 0.5, 0.5], 0)
        pennylane_circuit.U3([0.5, 0.5, 0.5], 0)
        qiskit_circuit.U3([0.5, 0.5, 0.5], 0)
        tket_circuit.U3([0.5, 0.5, 0.5], 0)

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CX gate
        cirq_circuit.CX(0, 1)
        pennylane_circuit.CX(0, 1)
        qiskit_circuit.CX(0, 1)
        tket_circuit.CX(0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CY gate
        cirq_circuit.CY(0, 1)
        pennylane_circuit.CY(0, 1)
        qiskit_circuit.CY(0, 1)
        tket_circuit.CY(0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CZ gate
        cirq_circuit.CZ(0, 1)
        pennylane_circuit.CZ(0, 1)
        qiskit_circuit.CZ(0, 1)
        tket_circuit.CZ(0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CH gate
        cirq_circuit.CH(0, 1)
        pennylane_circuit.CH(0, 1)
        qiskit_circuit.CH(0, 1)
        tket_circuit.CH(0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CS gate
        cirq_circuit.CS(0, 1)
        pennylane_circuit.CS(0, 1)
        qiskit_circuit.CS(0, 1)
        tket_circuit.CS(0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CT(self) -> None:
        """ Test the Controlled T gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CT gate
        cirq_circuit.CT(0, 1)
        pennylane_circuit.CT(0, 1)
        qiskit_circuit.CT(0, 1)
        tket_circuit.CT(0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CRX gate
        cirq_circuit.CRX(0.5, 0, 1)
        pennylane_circuit.CRX(0.5, 0, 1)
        qiskit_circuit.CRX(0.5, 0, 1)
        tket_circuit.CRX(0.5, 0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CRY gate
        cirq_circuit.CRY(0.5, 0, 1)
        pennylane_circuit.CRY(0.5, 0, 1)
        qiskit_circuit.CRY(0.5, 0, 1)
        tket_circuit.CRY(0.5, 0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CRZ gate
        cirq_circuit.CRZ(0.5, 0, 1)
        pennylane_circuit.CRZ(0.5, 0, 1)
        qiskit_circuit.CRZ(0.5, 0, 1)
        tket_circuit.CRZ(0.5, 0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(2, 2)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(2, 2)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(2, 2)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(2, 2)

        # Apply the CU3 gate
        cirq_circuit.CU3([0.5, 0.5, 0.5], 0, 1)
        pennylane_circuit.CU3([0.5, 0.5, 0.5], 0, 1)
        qiskit_circuit.CU3([0.5, 0.5, 0.5], 0, 1)
        tket_circuit.CU3([0.5, 0.5, 0.5], 0, 1)

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCX gate
        cirq_circuit.MCX([0, 1], [2, 3])
        pennylane_circuit.MCX([0, 1], [2, 3])
        qiskit_circuit.MCX([0, 1], [2, 3])
        tket_circuit.MCX([0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(8) == pennylane_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == qiskit_circuit.get_unitary().round(8))
        assert np.all(cirq_circuit.get_unitary().round(8) == tket_circuit.get_unitary().round(8))

    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCY gate
        cirq_circuit.MCY([0, 1], [2, 3])
        pennylane_circuit.MCY([0, 1], [2, 3])
        qiskit_circuit.MCY([0, 1], [2, 3])
        tket_circuit.MCY([0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.all(cirq_circuit.get_unitary().round(6) == pennylane_circuit.get_unitary().round(6))
        assert np.all(cirq_circuit.get_unitary().round(6) == qiskit_circuit.get_unitary().round(6))
        assert np.all(cirq_circuit.get_unitary().round(6) == tket_circuit.get_unitary().round(6))

    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCZ gate
        cirq_circuit.MCZ([0, 1], [2, 3])
        pennylane_circuit.MCZ([0, 1], [2, 3])
        qiskit_circuit.MCZ([0, 1], [2, 3])
        tket_circuit.MCZ([0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCH gate
        cirq_circuit.MCH([0, 1], [2, 3])
        pennylane_circuit.MCH([0, 1], [2, 3])
        qiskit_circuit.MCH([0, 1], [2, 3])
        tket_circuit.MCH([0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCS gate
        cirq_circuit.MCS([0, 1], [2, 3])
        pennylane_circuit.MCS([0, 1], [2, 3])
        qiskit_circuit.MCS([0, 1], [2, 3])
        tket_circuit.MCS([0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCT(self) -> None:
        """ Test the Multi-Controlled T gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCT gate
        cirq_circuit.MCT([0, 1], [2, 3])
        pennylane_circuit.MCT([0, 1], [2, 3])
        qiskit_circuit.MCT([0, 1], [2, 3])
        tket_circuit.MCT([0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCRX gate
        cirq_circuit.MCRX(0.5, [0, 1], [2, 3])
        pennylane_circuit.MCRX(0.5, [0, 1], [2, 3])
        qiskit_circuit.MCRX(0.5, [0, 1], [2, 3])
        tket_circuit.MCRX(0.5, [0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCRY gate
        cirq_circuit.MCRY(0.5, [0, 1], [2, 3])
        pennylane_circuit.MCRY(0.5, [0, 1], [2, 3])
        qiskit_circuit.MCRY(0.5, [0, 1], [2, 3])
        tket_circuit.MCRY(0.5, [0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCRZ gate
        cirq_circuit.MCRZ(0.5, [0, 1], [2, 3])
        pennylane_circuit.MCRZ(0.5, [0, 1], [2, 3])
        qiskit_circuit.MCRZ(0.5, [0, 1], [2, 3])
        tket_circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """
        # Define the `qickit.CirqCircuit` instance
        cirq_circuit = CirqCircuit(4, 4)

        # Define the `qickit.PennylaneCircuit` instance
        pennylane_circuit = PennylaneCircuit(4, 4)

        # Define the `qickit.QiskitCircuit` instance
        qiskit_circuit = QiskitCircuit(4, 4)

        # Define the `qickit.TKETCircuit` instance
        tket_circuit = TKETCircuit(4, 4)

        # Apply the MCU3 gate
        cirq_circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])
        pennylane_circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])
        qiskit_circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])
        tket_circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])

        # Qiskit uses LSB convention, and other frameworks use MSB
        qiskit_circuit.vertical_reverse()

        # Ensure they are equivalent
        assert np.allclose(cirq_circuit.get_unitary().round(6), pennylane_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), qiskit_circuit.get_unitary().round(6))
        assert np.allclose(cirq_circuit.get_unitary().round(6), tket_circuit.get_unitary().round(6))

    def test_measure(self) -> None:
        pass