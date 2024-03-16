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

__all__ = ['TestCirqCircuit']

import numpy as np

# Cirq imports
import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, CX, CZ
CY = cirq.ControlledGate(Y)

# QICKIT imports
from qickit.circuit import CirqCircuit
from tests.circuit import Template


class TestCirqCircuit(Template):
    """ `qickit.TestCirqCircuit` is the tester class for `qickit.CirqCircuit` class.
    """
    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Pauli X gate
        circuit.X(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(X(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Pauli Y gate
        circuit.Y(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Y(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Pauli Z gate
        circuit.Z(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Z(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(H(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Clifford S gate
        circuit.S(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(S(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Clifford T gate
        circuit.T(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(T(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Rx(rads=0.5)(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Ry(rads=0.5)(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Rz(rads=0.5)(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.5, 0.5, 0.5], 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)

        # Create a single qubit unitary gate
        u3 = [[np.cos(0.5/2), -np.exp(1j*0.5) * np.sin(0.5/2)],
              [np.exp(1j*0.5) * np.sin(0.5/2), np.exp(1j*(0.5 + 0.5)) * np.cos(0.5/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        cirq_circuit = cirq.Circuit(U3().on(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(CX(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(CY(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(CZ(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(H(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(S(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CT(self) -> None:
        """ Test the Controlled Clifford-T gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(T(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(Rx(rads=0.5)(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(Ry(rads=0.5)(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(Rz(rads=0.5)(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.5, 0.5, 0.5], 0, 1)

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)

        # Create a single qubit unitary gate
        u3 = [[np.cos(0.5/2), -np.exp(1j*0.5) * np.sin(0.5/2)],
              [np.exp(1j*0.5) * np.sin(0.5/2), np.exp(1j*(0.5 + 0.5)) * np.cos(0.5/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        cirq_circuit = cirq.Circuit(U3().on(qr[1]).controlled_by(qr[0]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(X(qr[2]).controlled_by(qr[0], qr[1]),
                                    X(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(Y(qr[2]).controlled_by(qr[0], qr[1]),
                                    Y(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(Z(qr[2]).controlled_by(qr[0], qr[1]),
                                    Z(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(H(qr[2]).controlled_by(qr[0], qr[1]),
                                    H(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(S(qr[2]).controlled_by(qr[0], qr[1]),
                                    S(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCT(self) -> None:
        """ Test the Multi-Controlled Clifford-T gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(T(qr[2]).controlled_by(qr[0], qr[1]),
                                    T(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(Rx(rads=0.5)(qr[2]).controlled_by(qr[0], qr[1]),
                                    Rx(rads=0.5)(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(Ry(rads=0.5)(qr[2]).controlled_by(qr[0], qr[1]),
                                    Ry(rads=0.5)(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.allclose(circuit.get_unitary(), cirq.unitary(cirq_circuit))

    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cirq_circuit = cirq.Circuit(Rz(rads=0.5)(qr[2]).controlled_by(qr[0], qr[1]),
                                    Rz(rads=0.5)(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])

        # `qickit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)

        # Create a single qubit unitary gate
        u3 = [[np.cos(0.5/2), -np.exp(1j*0.5) * np.sin(0.5/2)],
              [np.exp(1j*0.5) * np.sin(0.5/2), np.exp(1j*(0.5 + 0.5)) * np.cos(0.5/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        cirq_circuit = cirq.Circuit(U3().on(qr[2]).controlled_by(qr[0], qr[1]),
                                    U3().on(qr[3]).controlled_by(qr[0], qr[1]))

        assert np.all(circuit.get_unitary() == cirq.unitary(cirq_circuit))

    def test_measure(self) -> None:
        """ Test the measurement gate.
        """
        # Define the `qickit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the measurement gate
        circuit.measure(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)

        assert circuit.circuit == cirq.Circuit(cirq.measure(qr[0], key='meas'))