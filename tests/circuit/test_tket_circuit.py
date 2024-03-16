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

__all__ = ['TestTKETCircuit']

import numpy as np

# TKET imports
from pytket import Circuit as TKCircuit
from pytket import OpType
from pytket.circuit import Op, QControlBox

# QICKIT imports
from qickit.circuit import TKETCircuit
from tests.circuit import Template


class TestTKETCircuit(Template):
    """ `qickit.TestTKETCircuit` is the tester class for `qickit.TKETCircuit` class.
    """
    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the Pauli X gate
        circuit.X(0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.X, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the Pauli Y gate
        circuit.Y(0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Y, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the Pauli Z gate
        circuit.Z(0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Z, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.H, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the Clifford-S gate
        circuit.S(0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.S, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the Clifford-T gate
        circuit.T(0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.T, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Rx, 0.5/np.pi, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Ry, 0.5/np.pi, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Rz, 0.5/np.pi, [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.5, 0.5, 0.5], 0)

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.U3, [0.5/np.pi, 0.5/np.pi, 0.5/np.pi], [0])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CX, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CY, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CZ, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CH, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CS, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CT(self) -> None:
        """ Test the Controlled Clifford-T gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        t = Op.create(OpType.T)
        ct = QControlBox(t, 1)
        tket_circuit.add_gate(ct, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        crx = OpType.CRx
        tket_circuit.add_gate(crx, 0.5/np.pi, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        cry = OpType.CRy
        tket_circuit.add_gate(cry, 0.5/np.pi, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        crz = OpType.CRz
        tket_circuit.add_gate(crz, 0.5/np.pi, [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.5, 0.5, 0.5], 0, 1)

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        cu3 = OpType.CU3
        tket_circuit.add_gate(cu3, [0.5/np.pi, 0.5/np.pi, 0.5/np.pi], [0, 1])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        tket_circuit.add_gate(OpType.CnX, [0, 1, 2])
        tket_circuit.add_gate(OpType.CnX, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        tket_circuit.add_gate(OpType.CnY, [0, 1, 2])
        tket_circuit.add_gate(OpType.CnY, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        tket_circuit.add_gate(OpType.CnZ, [0, 1, 2])
        tket_circuit.add_gate(OpType.CnZ, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        h = Op.create(OpType.H)
        ch = QControlBox(h, 2)
        tket_circuit.add_gate(ch, [0, 1, 2])
        tket_circuit.add_gate(ch, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        s = Op.create(OpType.S)
        cs = QControlBox(s, 2)
        tket_circuit.add_gate(cs, [0, 1, 2])
        tket_circuit.add_gate(cs, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCT(self) -> None:
        """ Test the Multi-Controlled Clifford-T gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        t = Op.create(OpType.T)
        ct = QControlBox(t, 2)
        tket_circuit.add_gate(ct, [0, 1, 2])
        tket_circuit.add_gate(ct, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        rx = Op.create(OpType.Rx, 0.5/np.pi)
        crx = QControlBox(rx, 2)
        tket_circuit.add_gate(crx, [0, 1, 2])
        tket_circuit.add_gate(crx, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        ry = Op.create(OpType.Ry, 0.5/np.pi)
        cry = QControlBox(ry, 2)
        tket_circuit.add_gate(cry, [0, 1, 2])
        tket_circuit.add_gate(cry, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        rz = Op.create(OpType.Rz, 0.5/np.pi)
        crz = QControlBox(rz, 2)
        tket_circuit.add_gate(crz, [0, 1, 2])
        tket_circuit.add_gate(crz, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """
        # Define the `qickit.TKETCircuit` instance
        circuit = TKETCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])

        # `qickit.TKETCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(4, 4)
        u3 = Op.create(OpType.U3, [0.5/np.pi, 0.5/np.pi, 0.5/np.pi])
        cu3 = QControlBox(u3, 2)
        tket_circuit.add_gate(cu3, [0, 1, 2])
        tket_circuit.add_gate(cu3, [0, 1, 3])

        assert np.all(circuit.get_unitary() == tket_circuit.get_unitary())

    def test_measure(self) -> None:
        return super().test_measure()