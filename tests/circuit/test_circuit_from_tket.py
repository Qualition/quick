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

from pytket import Circuit as TKCircuit
from pytket import OpType
from pytket.circuit import Op, QControlBox

from qickit.circuit import Circuit, TKETCircuit
from tests.circuit import FrameworkTemplate


class TestFromTKET(FrameworkTemplate):
    def test_Identity(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.noop, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.Identity([0])
        assert qickit_circuit == check_circuit

    def test_X(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.X, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)
        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.X([0])
        assert qickit_circuit == check_circuit

    def test_Y(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Y, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.Y([0])
        assert qickit_circuit == check_circuit

    def test_Z(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Z, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.Z([0])
        assert qickit_circuit == check_circuit

    def test_H(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.H, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.H([0])
        assert qickit_circuit == check_circuit

    def test_S(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.S, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.S([0])
        assert qickit_circuit == check_circuit

    def test_Sdg(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Sdg, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.Sdg([0])
        assert qickit_circuit == check_circuit

    def test_T(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.T, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.T([0])
        assert qickit_circuit == check_circuit

    def test_Tdg(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Tdg, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.Tdg([0])
        assert qickit_circuit == check_circuit

    def test_RX(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Rx, 0.5, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)
        print(qickit_circuit)
        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.RX(0.5, 0)
        print(check_circuit)
        assert qickit_circuit == check_circuit

    def test_RY(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Ry, 0.5, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.RY(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_RZ(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.add_gate(OpType.Rz, 0.5, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.RZ(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_U3(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(1, 1)
        angles = [0.1, 0.2, 0.3]
        tket_circuit.add_gate(OpType.U3, angles, [0])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(1)
        check_circuit.U3(angles, 0)
        assert qickit_circuit == check_circuit

    def test_SWAP(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.SWAP, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.SWAP(0, 1)
        assert qickit_circuit == check_circuit

    def test_CX(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CX, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CX(0, 1)
        assert qickit_circuit == check_circuit

    def test_CY(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CY, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CY(0, 1)
        assert qickit_circuit == check_circuit

    def test_CZ(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CZ, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CZ(0, 1)
        assert qickit_circuit == check_circuit

    def test_CH(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CH, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CH(0, 1)
        assert qickit_circuit == check_circuit

    def test_CS(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CS, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CS(0, 1)
        assert qickit_circuit == check_circuit

    def test_CSdg(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CSdg, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CSdg(0, 1)
        assert qickit_circuit == check_circuit

    def test_CT(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        t = Op.create(OpType.T)
        ct = QControlBox(t, 1)
        tket_circuit.add_qcontrolbox(ct, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CT(0, 1)
        assert qickit_circuit == check_circuit

    def test_CTdg(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tdg = Op.create(OpType.Tdg)
        ctdg = QControlBox(tdg, 1)
        tket_circuit.add_qcontrolbox(ctdg, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CTdg(0, 1)
        assert qickit_circuit == check_circuit

    def test_CRX(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CRx, 0.5, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CRX(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CRY(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CRy, 0.5, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CRY(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CRZ(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        tket_circuit.add_gate(OpType.CRz, 0.5, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CRZ(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CU3(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(2, 2)
        angles = [0.1, 0.2, 0.3]
        tket_circuit.add_gate(OpType.CU3, angles, [0, 1])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(2)
        check_circuit.CU3(angles, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CSWAP(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(3, 3)
        tket_circuit.add_gate(OpType.CSWAP, [0, 1, 2])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)
        print(qickit_circuit)
        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(3)
        check_circuit.CSWAP(0, 1, 2)
        print(check_circuit)
        assert qickit_circuit == check_circuit

    def test_MCX(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        tket_circuit.add_gate(OpType.CnX, [0, 1, 2])
        tket_circuit.add_gate(OpType.CnX, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCX([0, 1], 2)
        check_circuit.MCX([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCY(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        tket_circuit.add_gate(OpType.CnY, [0, 1, 2])
        tket_circuit.add_gate(OpType.CnY, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCY([0, 1], 2)
        check_circuit.MCY([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCZ(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        tket_circuit.add_gate(OpType.CnZ, [0, 1, 2])
        tket_circuit.add_gate(OpType.CnZ, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCZ([0, 1], 2)
        check_circuit.MCZ([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCH(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        h = Op.create(OpType.H)
        mch = QControlBox(h, 2)
        tket_circuit.add_qcontrolbox(mch, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mch, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCH([0, 1], 2)
        check_circuit.MCH([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCS(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        s = Op.create(OpType.S)
        mcs = QControlBox(s, 2)
        tket_circuit.add_qcontrolbox(mcs, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mcs, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCS([0, 1], 2)
        check_circuit.MCS([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCSdg(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        sdg = Op.create(OpType.Sdg)
        mcsdg = QControlBox(sdg, 2)
        tket_circuit.add_qcontrolbox(mcsdg, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mcsdg, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCSdg([0, 1], 2)
        check_circuit.MCSdg([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCT(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        t = Op.create(OpType.T)
        mct = QControlBox(t, 2)
        tket_circuit.add_qcontrolbox(mct, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mct, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCT([0, 1], 2)
        check_circuit.MCT([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCTdg(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        tdg = Op.create(OpType.Tdg)
        mctdg = QControlBox(tdg, 2)
        tket_circuit.add_qcontrolbox(mctdg, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mctdg, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCTdg([0, 1], 2)
        check_circuit.MCTdg([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRX(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        rx = Op.create(OpType.Rx, 0.5)
        mcrx = QControlBox(rx, 2)
        tket_circuit.add_qcontrolbox(mcrx, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mcrx, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCRX(0.5, [0, 1], 2)
        check_circuit.MCRX(0.5, [0, 1], 3)

        assert qickit_circuit == check_circuit

    def test_MCRY(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        ry = Op.create(OpType.Ry, 0.5)
        mcry = QControlBox(ry, 2)
        tket_circuit.add_qcontrolbox(mcry, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mcry, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCRY(0.5, [0, 1], 2)
        check_circuit.MCRY(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRZ(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        rz = Op.create(OpType.Rz, 0.5)
        mcrz = QControlBox(rz, 2)
        tket_circuit.add_qcontrolbox(mcrz, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mcrz, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCRZ(0.5, [0, 1], 2)
        check_circuit.MCRZ(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCU3(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        angles = [0.1, 0.2, 0.3]
        u3 = Op.create(OpType.U3, angles)
        mcu3 = QControlBox(u3, 2)
        tket_circuit.add_qcontrolbox(mcu3, [0, 1, 2])
        tket_circuit.add_qcontrolbox(mcu3, [0, 1, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCU3(angles, [0, 1], 2)
        check_circuit.MCU3(angles, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCSWAP(self) -> None:
        # Define the TKET circuit
        tket_circuit = TKCircuit(4, 4)
        swap = Op.create(OpType.SWAP)
        mcswap = QControlBox(swap, 2)
        tket_circuit.add_gate(mcswap, [0, 1, 2, 3])

        # Convert the TKET circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)
        print(qickit_circuit)
        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = TKETCircuit(4)
        check_circuit.MCSWAP([0, 1], 2, 3)
        print(check_circuit)
        assert qickit_circuit == check_circuit

    def test_GlobalPhase(self) -> None:
        # # Define the TKET circuit
        # tket_circuit = TKCircuit(1, 1)
        # tket_circuit.add_phase(0.5)

        # # Convert the TKET circuit to a QICKIT circuit
        # qickit_circuit = Circuit.from_tket(tket_circuit, TKETCircuit)

        # # Define the equivalent QICKIT circuit, and ensure
        # # that the two circuits are equal
        # check_circuit = TKETCircuit(1)
        # check_circuit.GlobalPhase(0.5)
        # assert qickit_circuit == check_circuit
        pass

    def test_single_measurement(self) -> None:
        pass

    def test_multiple_measurement(self) -> None:
        pass