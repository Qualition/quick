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

__all__ = ['TestQiskitCircuit']

import numpy as np
from numpy.testing import assert_almost_equal

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator

# QICKIT imports
from qickit.circuit import QiskitCircuit
from tests.circuit import Template


class TestQiskitCircuit(Template):
    """ `qickit.TestQiskitCircuit` is the tester class for `qickit.QiskitCircuit` class.
    """
    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the Pauli X gate
        circuit.X(0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.x(0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the Pauli Y gate
        circuit.Y(0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.y(0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the Pauli Z gate
        circuit.Z(0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.z(0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.h(0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the S gate
        circuit.S(0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.s(0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the T gate
        circuit.T(0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.t(0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rx(0.5, 0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.ry(0.5, 0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rz(0.5, 0)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.5, 0.5, 0.5], 0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.append(U3Gate(0.5, 0.5, 0.5), [0])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cx(0, 1)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cy(0, 1)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cz(0, 1)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.ch(0, 1)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cs(0, 1)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CT(self) -> None:
        """ Test the Controlled Clifford-T gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        ct = TGate().control(1)
        qiskit_circuit.append(ct, [0, 1])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        crx = RXGate(0.5).control(1)
        qiskit_circuit.append(crx, [0, 1])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        cry = RYGate(0.5).control(1)
        qiskit_circuit.append(cry, [0, 1])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        crz = RZGate(0.5).control(1)
        qiskit_circuit.append(crz, [0, 1])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.5, 0.5, 0.5], 0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        cu3 = U3Gate(0.5, 0.5, 0.5).control(1)
        qiskit_circuit.append(cu3, [0, 1])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcx = XGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcx, [0, 1, 2])
        qiskit_circuit.append(mcx, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcy = YGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcy, [0, 1, 2])
        qiskit_circuit.append(mcy, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcz = ZGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcz, [0, 1, 2])
        qiskit_circuit.append(mcz, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mch = HGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mch, [0, 1, 2])
        qiskit_circuit.append(mch, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcs = SGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcs, [0, 1, 2])
        qiskit_circuit.append(mcs, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCT(self) -> None:
        """ Test the Multi-Controlled Clifford-T gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mct = TGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mct, [0, 1, 2])
        qiskit_circuit.append(mct, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcrx = RXGate(0.5).control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcrx, [0, 1, 2])
        qiskit_circuit.append(mcrx, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcry = RYGate(0.5).control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcry, [0, 1, 2])
        qiskit_circuit.append(mcry, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcrz = RZGate(0.5).control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcrz, [0, 1, 2])
        qiskit_circuit.append(mcrz, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """
        # Define the `qickit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcu3 = U3Gate(0.5, 0.5, 0.5).control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcu3, [0, 1, 2])
        qiskit_circuit.append(mcu3, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_measure(self) -> None:
        return super().test_measure()