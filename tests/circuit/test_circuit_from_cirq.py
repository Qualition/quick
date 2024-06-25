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

import numpy as np

# Cirq imports
import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, SWAP, I

# QICKIT imports
from qickit.circuit import Circuit, QiskitCircuit
from tests.circuit import FrameworkTemplate


class TestFromCirq(FrameworkTemplate):
    """ `tests.circuit.TestFromCirq` tests the `.from_cirq` method.
    """
    def test_Identity(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(I(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.Identity(0)
        assert qickit_circuit == check_circuit

    def test_X(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(X(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.X(0)
        assert qickit_circuit == check_circuit

    def test_Y(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(Y(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.Y(0)
        assert qickit_circuit == check_circuit

    def test_Z(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(Z(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.Z(0)
        assert qickit_circuit == check_circuit

    def test_H(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(H(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.H(0)
        assert qickit_circuit == check_circuit

    def test_S(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(S(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.S(0)
        assert qickit_circuit == check_circuit

    def test_T(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(T(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.T(0)
        assert qickit_circuit == check_circuit

    def test_RX(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        rx = Rx(rads=0.5)
        cirq_circuit.append(rx(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.RX(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_RY(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        ry = Ry(rads=0.5)
        cirq_circuit.append(ry(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.RY(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_RZ(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(1)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        rz = Rz(rads=0.5)
        cirq_circuit.append(rz(qr[0]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.RZ(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_U3(self) -> None:
        # # Create a single qubit unitary gate
        # angles = [0.1, 0.2, 0.3]
        # u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
        #       [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
        #                                                    np.cos(angles[0]/2)]]

        # # Define the U3 gate class
        # class U3(cirq.Gate):
        #     def __init__(self):
        #         super(U3, self)

        #     def _num_qubits_(self):
        #         return 1

        #     def _unitary_(self):
        #         return np.array(u3)

        #     def _circuit_diagram_info_(self, args):
        #         return "U3"

        # # Define the quantum bit register
        # qr = cirq.LineQubit.range(1)

        # # Define the Cirq circuit
        # cirq_circuit = cirq.Circuit()
        # cirq_circuit.append(U3().on(qr[0]))

        # # Convert the Cirq circuit to a QICKIT circuit
        # qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # # Define the equivalent QICKIT circuit, and ensure
        # # that the two circuits are equal
        # check_circuit = QiskitCircuit(1)
        # check_circuit.U3(angles, 0)
        # assert qickit_circuit == check_circuit
        pass

    def test_SWAP(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(SWAP(qr[0], qr[1]))

        for operation in list(cirq_circuit.all_operations()):
            gate = operation.gate
            gate_type = type(gate).__name__
            print(gate_type)

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.SWAP(0, 1)
        assert qickit_circuit == check_circuit

    def test_CX(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cx = cirq.ControlledGate(sub_gate=X, num_controls=1)
        cirq_circuit.append(cx(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CX(0, 1)
        assert qickit_circuit == check_circuit

    def test_CY(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cy = cirq.ControlledGate(sub_gate=Y, num_controls=1)
        cirq_circuit.append(cy(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CY(0, 1)
        assert qickit_circuit == check_circuit

    def test_CZ(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cz = cirq.ControlledGate(sub_gate=Z, num_controls=1)
        cirq_circuit.append(cz(qr[0], qr[1]))

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit(cz(cirq.LineQubit(0), cirq.LineQubit(1)))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CZ(0, 1)
        assert qickit_circuit == check_circuit

    def test_CH(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        ch = cirq.ControlledGate(sub_gate=H, num_controls=1)
        cirq_circuit.append(ch(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CH(0, 1)
        assert qickit_circuit == check_circuit

    def test_CS(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cs = cirq.ControlledGate(sub_gate=S, num_controls=1)
        cirq_circuit.append(cs(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CS(0, 1)
        assert qickit_circuit == check_circuit

    def test_CT(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        ct = cirq.ControlledGate(sub_gate=T, num_controls=1)
        cirq_circuit.append(ct(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CT(0, 1)
        assert qickit_circuit == check_circuit

    def test_CRX(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        crx = cirq.ControlledGate(sub_gate=Rx(rads=0.5), num_controls=1)
        cirq_circuit.append(crx(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CRX(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CRY(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cry = cirq.ControlledGate(sub_gate=Ry(rads=0.5), num_controls=1)
        cirq_circuit.append(cry(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CRY(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CRZ(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(2)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        crz = cirq.ControlledGate(sub_gate=Rz(rads=0.5), num_controls=1)
        cirq_circuit.append(crz(qr[0], qr[1]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CRZ(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CU3(self) -> None:
        # # Create a single qubit unitary gate
        # angles = [0.1, 0.2, 0.3]
        # u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
        #       [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
        #                                                    np.cos(angles[0]/2)]]

        # # Define the U3 gate class
        # class U3(cirq.Gate):
        #     def __init__(self):
        #         super(U3, self)

        #     def _num_qubits_(self):
        #         return 1

        #     def _unitary_(self):
        #         return np.array(u3)

        #     def _circuit_diagram_info_(self, args):
        #         return "U3"

        # # Define the quantum bit register
        # qr = cirq.LineQubit.range(2)

        # # Define the Cirq circuit
        # cirq_circuit = cirq.Circuit()
        # cu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=1)
        # cirq_circuit = cirq_circuit.append(cu3(qr[0], qr[1]))

        # # Convert the Cirq circuit to a QICKIT circuit
        # qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # # Define the equivalent QICKIT circuit, and ensure
        # # that the two circuits are equal
        # check_circuit = QiskitCircuit(2)
        # check_circuit.CU3(angles, 0, 1)
        # assert qickit_circuit == check_circuit
        pass

    def test_CSWAP(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(3)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        cswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=1)
        cirq_circuit.append(cswap(qr[0], qr[1], qr[2]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(3)
        check_circuit.CSWAP(0, 1, 2)
        assert qickit_circuit == check_circuit

    def test_MCX(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcx = cirq.ControlledGate(sub_gate=X, num_controls=2)
        cirq_circuit.append(mcx(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcx(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)
        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCX([0, 1], 2)
        check_circuit.MCX([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCY(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcy = cirq.ControlledGate(sub_gate=Y, num_controls=2)
        cirq_circuit.append(mcy(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcy(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCY([0, 1], 2)
        check_circuit.MCY([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCZ(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcz = cirq.ControlledGate(sub_gate=Z, num_controls=2)
        cirq_circuit.append(mcz(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcz(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCZ([0, 1], 2)
        check_circuit.MCZ([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCH(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mch = cirq.ControlledGate(sub_gate=H, num_controls=2)
        cirq_circuit.append(mch(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mch(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCH([0, 1], 2)
        check_circuit.MCH([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCS(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcs = cirq.ControlledGate(sub_gate=S, num_controls=2)
        cirq_circuit.append(mcs(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcs(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCS([0, 1], 2)
        check_circuit.MCS([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCT(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mct = cirq.ControlledGate(sub_gate=T, num_controls=2)
        cirq_circuit.append(mct(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mct(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCT([0, 1], 2)
        check_circuit.MCT([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRX(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcrx = cirq.ControlledGate(sub_gate=Rx(rads=0.5), num_controls=2)
        cirq_circuit.append(mcrx(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcrx(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCRX(0.5, [0, 1], 2)
        check_circuit.MCRX(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRY(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcry = cirq.ControlledGate(sub_gate=Ry(rads=0.5), num_controls=2)
        cirq_circuit.append(mcry(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcry(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCRY(0.5, [0, 1], 2)
        check_circuit.MCRY(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRZ(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcrz = cirq.ControlledGate(sub_gate=Rz(rads=0.5), num_controls=2)
        cirq_circuit.append(mcrz(qr[0], qr[1], qr[2]))
        cirq_circuit.append(mcrz(qr[0], qr[1], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCRZ(0.5, [0, 1], 2)
        check_circuit.MCRZ(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCU3(self) -> None:
        # # Create a single qubit unitary gate
        # angles = [0.1, 0.2, 0.3]
        # u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
        #       [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
        #                                                    np.cos(angles[0]/2)]]

        # # Define the U3 gate class
        # class U3(cirq.Gate):
        #     def __init__(self):
        #         super(U3, self)

        #     def _num_qubits_(self):
        #         return 1

        #     def _unitary_(self):
        #         return np.array(u3)

        #     def _circuit_diagram_info_(self, args):
        #         return "U3"

        # # Define the quantum bit register
        # qr = cirq.LineQubit.range(4)

        # # Define the Cirq circuit
        # cirq_circuit = cirq.Circuit()
        # mcu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=2)
        # cirq_circuit.append(mcu3(qr[0], qr[1], qr[2]))
        # cirq_circuit.append(mcu3(qr[0], qr[1], qr[3]))

        # # Convert the Cirq circuit to a QICKIT circuit
        # qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # # Define the equivalent QICKIT circuit, and ensure
        # # that the two circuits are equal
        # check_circuit = QiskitCircuit(4)
        # check_circuit.MCU3(angles, [0, 1], [2, 3])
        # assert qickit_circuit == check_circuit
        pass

    def test_MCSWAP(self) -> None:
        # Define the quantum bit register
        qr = cirq.LineQubit.range(4)

        # Define the Cirq circuit
        cirq_circuit = cirq.Circuit()
        mcswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=2)
        cirq_circuit.append(mcswap(qr[0], qr[1], qr[2], qr[3]))

        # Convert the Cirq circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCSWAP([0, 1], 2, 3)
        assert qickit_circuit == check_circuit

    def test_GlobalPhase(self) -> None:
        # # Define the Cirq circuit
        # cirq_circuit = cirq.Circuit()
        # global_phase = cirq.GlobalPhaseGate(np.exp(1j*0.5))
        # cirq_circuit.append(global_phase())

        # # Convert the Cirq circuit to a QICKIT circuit
        # qickit_circuit = Circuit.from_cirq(cirq_circuit, QiskitCircuit)

        # # Define the equivalent QICKIT circuit, and ensure
        # # that the two circuits are equal
        # check_circuit = QiskitCircuit(1)
        # check_circuit.GlobalPhase(0.5)
        # assert qickit_circuit == check_circuit
        pass

    def test_single_measurement(self) -> None:
        pass

    def test_multiple_measurement(self) -> None:
        pass