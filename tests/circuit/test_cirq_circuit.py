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

import cirq.ops.identity

__all__ = ['TestCirqCircuit']

import cirq.ops
import numpy as np
from numpy.testing import assert_almost_equal

# Cirq imports
import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, SWAP, I

# QICKIT imports
from qickit.circuit import CirqCircuit
from tests.circuit import Template


class TestCirqCircuit(Template):
    """ `tests.circuit.TestCirqCircuit` is the tester class for `qickit.circuit.CirqCircuit` class.
    """
    def test_X(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Pauli X gate
        circuit.X(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(X(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_Y(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Pauli Y gate
        circuit.Y(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Y(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_Z(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Pauli Z gate
        circuit.Z(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Z(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_H(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(H(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_S(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Clifford S gate
        circuit.S(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(S(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_T(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the Clifford T gate
        circuit.T(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(T(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_RX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Rx(rads=0.5)(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_RY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Ry(rads=0.5)(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_RZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(Rz(rads=0.5)(qr[0]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_U3(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the U3 gate
        params = [0.1, 0.2, 0.3]
        circuit.U3(params, 0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)

        # Create a single qubit unitary gate
        u3 = [[np.cos(params[0]/2), -np.exp(1j*params[2]) * np.sin(params[0]/2)],
              [np.exp(1j*params[1]) * np.sin(params[0]/2), np.exp(1j*(params[1] + params[2])) * \
                                                           np.cos(params[0]/2)]]

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

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_SWAP(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the SWAP gate
        circuit.SWAP(0, 1)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit(cirq.SWAP(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cx = cirq.ControlledGate(sub_gate=X, num_controls=1)
        cirq_circuit = cirq.Circuit(cx(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cy = cirq.ControlledGate(sub_gate=Y, num_controls=1)
        cirq_circuit = cirq.Circuit(cy(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cz = cirq.ControlledGate(sub_gate=Z, num_controls=1)
        cirq_circuit = cirq.Circuit(cz(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CH(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        ch = cirq.ControlledGate(sub_gate=H, num_controls=1)
        cirq_circuit = cirq.Circuit(ch(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CS(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cs = cirq.ControlledGate(sub_gate=S, num_controls=1)
        cirq_circuit = cirq.Circuit(cs(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CT(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        ct = cirq.ControlledGate(sub_gate=T, num_controls=1)
        cirq_circuit = cirq.Circuit(ct(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CRX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        crx = cirq.ControlledGate(sub_gate=Rx(rads=0.5), num_controls=1)
        cirq_circuit = cirq.Circuit(crx(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CRY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        cry = cirq.ControlledGate(sub_gate=Ry(rads=0.5), num_controls=1)
        cirq_circuit = cirq.Circuit(cry(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CRZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)
        crz = cirq.ControlledGate(sub_gate=Rz(rads=0.5), num_controls=1)
        cirq_circuit = cirq.Circuit(crz(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CU3(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply the CU3 gate
        params = [0.1, 0.2, 0.3]
        circuit.CU3(params, 0, 1)

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(2)

        # Create a single qubit unitary gate
        u3 = [[np.cos(params[0]/2), -np.exp(1j*params[2]) * np.sin(params[0]/2)],
              [np.exp(1j*params[1]) * np.sin(params[0]/2), np.exp(1j*(params[1] + params[2])) * \
                                                           np.cos(params[0]/2)]]

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

        cu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=1)
        cirq_circuit = cirq.Circuit(cu3(qr[0], qr[1]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_CSWAP(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(3, 3)

        # Apply the CSWAP gate
        circuit.CSWAP(0, 1, 2)
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(3)
        cswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=1)
        cirq_circuit = cirq.Circuit(cswap(qr[0], qr[1], qr[2]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcx = cirq.ControlledGate(sub_gate=X, num_controls=2)
        cirq_circuit = cirq.Circuit(mcx(qr[0], qr[1], qr[2]),
                                    mcx(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcy = cirq.ControlledGate(sub_gate=Y, num_controls=2)
        cirq_circuit = cirq.Circuit(mcy(qr[0], qr[1], qr[2]),
                                    mcy(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcz = cirq.ControlledGate(sub_gate=Z, num_controls=2)
        cirq_circuit = cirq.Circuit(mcz(qr[0], qr[1], qr[2]),
                                    mcz(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCH(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mch = cirq.ControlledGate(sub_gate=H, num_controls=2)
        cirq_circuit = cirq.Circuit(mch(qr[0], qr[1], qr[2]),
                                    mch(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCS(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcs = cirq.ControlledGate(sub_gate=S, num_controls=2)
        cirq_circuit = cirq.Circuit(mcs(qr[0], qr[1], qr[2]),
                                    mcs(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCT(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mct = cirq.ControlledGate(sub_gate=T, num_controls=2)
        cirq_circuit = cirq.Circuit(mct(qr[0], qr[1], qr[2]),
                                    mct(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCRX(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcrx = cirq.ControlledGate(sub_gate=Rx(rads=0.5), num_controls=2)
        cirq_circuit = cirq.Circuit(mcrx(qr[0], qr[1], qr[2]),
                                    mcrx(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCRY(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcry = cirq.ControlledGate(sub_gate=Ry(rads=0.5), num_controls=2)
        cirq_circuit = cirq.Circuit(mcry(qr[0], qr[1], qr[2]),
                                    mcry(qr[0], qr[1], qr[3]))

        assert np.allclose(circuit.get_unitary(), cirq.unitary(cirq_circuit))

    def test_MCRZ(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        mcrz = cirq.ControlledGate(sub_gate=Rz(rads=0.5), num_controls=2)
        cirq_circuit = cirq.Circuit(mcrz(qr[0], qr[1], qr[2]),
                                    mcrz(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCU3(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCU3 gate
        params = [0.1, 0.2, 0.3]
        circuit.MCU3(params, [0, 1], [2, 3])

        # `qickit.circuit.CirqCircuit` uses LSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)

        # Create a single qubit unitary gate
        u3 = [[np.cos(params[0]/2), -np.exp(1j*params[2]) * np.sin(params[0]/2)],
              [np.exp(1j*params[1]) * np.sin(params[0]/2), np.exp(1j*(params[1] + params[2])) * \
                                                           np.cos(params[0]/2)]]

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

        mcu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=2)
        cirq_circuit = cirq.Circuit(mcu3(qr[0], qr[1], qr[2]),
                                    mcu3(qr[0], qr[1], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_MCSWAP(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCSWAP gate
        circuit.MCSWAP([0, 1], 2, 3)
        circuit.vertical_reverse()

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(4)
        cswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=2)
        cirq_circuit = cirq.Circuit(cswap(qr[0], qr[1], qr[2], qr[3]))

        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit), 8)

    def test_GlobalPhase(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(I(qr[0]))

        # Ensure the global phase is correct
        assert_almost_equal(circuit.get_unitary(), cirq.unitary(cirq_circuit) * np.exp(1j*1.8))

    def test_measure(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the measurement gate
        circuit.measure(0)

        # Define the equivalent `cirq.Circuit` instance, and
        # ensure they are equivalent
        qr = cirq.LineQubit.range(1)
        cirq_circuit = cirq.Circuit(I(qr[0]),
                                    cirq.measure(qr[0], key='meas'))

        assert circuit.circuit == cirq_circuit

    def test_unitary(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the gate
        circuit.MCX([0, 1], [2, 3])

        # Define the unitary
        unitary = circuit.get_unitary()

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        unitary_circuit = CirqCircuit(4, 4)
        unitary_circuit.unitary(unitary, [0, 1, 2, 3])

        assert_almost_equal(unitary_circuit.get_unitary(), unitary, 8)

    def test_vertical_reverse(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply GHZ state
        circuit.H(0)
        circuit.CX(0, 1)

        # Apply the vertical reverse operation
        circuit.vertical_reverse()

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = CirqCircuit(2, 2)
        updated_circuit.H(1)
        updated_circuit.CX(1, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_horizontal_reverse(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(2, 2)

        # Apply a RX and CX gate
        circuit.RX(np.pi, 0)
        circuit.CX(0, 1)

        # Apply the horizontal reverse operation
        circuit.horizontal_reverse()

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = CirqCircuit(2, 2)
        updated_circuit.CX(0, 1)
        updated_circuit.RX(-np.pi, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_add(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instances
        circuit1 = CirqCircuit(2, 2)
        circuit2 = CirqCircuit(2, 2)

        # Apply the Pauli-X gate
        circuit1.CX(0, 1)
        circuit2.CY(1, 0)

        # Add the two circuits
        circuit1.add(circuit2, [0, 1])

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        added_circuit = CirqCircuit(2, 2)
        added_circuit.CX(0, 1)
        added_circuit.CY(1, 0)

        assert circuit1 == added_circuit
        assert_almost_equal(circuit1.get_unitary(), added_circuit.get_unitary(), 8)

    def test_transpile(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        transpiled_circuit = CirqCircuit(4, 4)
        transpiled_circuit.MCX([0, 1], [2, 3])
        transpiled_circuit.transpile()

        assert_almost_equal(circuit.get_unitary(), transpiled_circuit.get_unitary(), 8)

    def test_get_depth(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Get the depth of the circuit, and ensure it is correct
        depth = circuit.get_depth()

        assert depth == 21

    def test_get_width(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Get the width of the circuit, and ensure it is correct
        width = circuit.get_width()

        assert width == 4

    def test_compress(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(1, 1)

        # Apply the MCX gate
        circuit.RX(np.pi/2, 0)

        # Compress the circuit
        circuit.compress(1.0)

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        compressed_circuit = CirqCircuit(1, 1)

        assert circuit == compressed_circuit
        assert_almost_equal(circuit.get_unitary(), compressed_circuit.get_unitary(), 8)

    def test_change_mapping(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = CirqCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Change the mapping of the circuit
        circuit.change_mapping([3, 2, 1, 0])

        # Define the equivalent `qickit.circuit.CirqCircuit` instance, and
        # ensure they are equivalent
        mapped_circuit = CirqCircuit(4, 4)
        mapped_circuit.MCX([3, 2], [1, 0])

        assert circuit == mapped_circuit
        assert_almost_equal(circuit.get_unitary(), mapped_circuit.get_unitary(), 8)