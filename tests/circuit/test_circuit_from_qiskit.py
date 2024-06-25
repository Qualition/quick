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

__all__ = ["TestFromQiskit"]

# Qiskit imports
from qiskit import QuantumCircuit # type: ignore
from qiskit.circuit.library import (RXGate, RYGate, RZGate, HGate, XGate, YGate, # type: ignore
                                    ZGate, SGate, TGate, U3Gate, SwapGate, # type: ignore
                                    GlobalPhaseGate) # type: ignore

# QICKIT imports
from qickit.circuit import Circuit, QiskitCircuit
from tests.circuit import FrameworkTemplate


class TestFromQiskit(FrameworkTemplate):
    """ `tests.circuit.TestFromQiskit` tests the `.from_qiskit` method.
    """
    def test_Identity(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.id(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.Identity(0)
        assert qickit_circuit == check_circuit

    def test_X(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.x(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.X(0)
        assert qickit_circuit == check_circuit

    def test_Y(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.y(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.Y(0)
        assert qickit_circuit == check_circuit

    def test_Z(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.z(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.Z(0)
        assert qickit_circuit == check_circuit

    def test_H(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.h(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.H(0)
        assert qickit_circuit == check_circuit

    def test_S(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.s(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.S(0)
        assert qickit_circuit == check_circuit

    def test_T(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.t(0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.T(0)
        assert qickit_circuit == check_circuit

    def test_RX(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rx(0.5, 0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.RX(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_RY(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.ry(0.5, 0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.RY(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_RZ(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.rz(0.5, 0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.RZ(0.5, 0)
        assert qickit_circuit == check_circuit

    def test_U3(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(1, 1)
        angles = [0.1, 0.2, 0.3]
        qiskit_circuit.u(angles[0], angles[1], angles[2], 0)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(1)
        check_circuit.U3(angles, 0)
        assert qickit_circuit == check_circuit

    def test_SWAP(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.swap(0, 1)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.SWAP(0, 1)
        assert qickit_circuit == check_circuit

    def test_CX(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cx(0, 1)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CX(0, 1)
        assert qickit_circuit == check_circuit

    def test_CY(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cy(0, 1)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CY(0, 1)
        assert qickit_circuit == check_circuit

    def test_CZ(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cz(0, 1)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CZ(0, 1)
        assert qickit_circuit == check_circuit

    def test_CH(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.ch(0, 1)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CH(0, 1)
        assert qickit_circuit == check_circuit

    def test_CS(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.cs(0, 1)

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CS(0, 1)
        assert qickit_circuit == check_circuit

    def test_CT(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        ct = TGate().control(1)
        qiskit_circuit.append(ct, [0, 1])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CT(0, 1)
        assert qickit_circuit == check_circuit

    def test_CRX(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        crx = RXGate(0.5).control(1)
        qiskit_circuit.append(crx, [0, 1])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CRX(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CRY(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        cry = RYGate(0.5).control(1)
        qiskit_circuit.append(cry, [0, 1])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CRY(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CRZ(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        crz = RZGate(0.5).control(1)
        qiskit_circuit.append(crz, [0, 1])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CRZ(0.5, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CU3(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(2, 2)
        angles = [0.1, 0.2, 0.3]
        cu3 = U3Gate(angles[0], angles[1], angles[2]).control(1)
        qiskit_circuit.append(cu3, [0, 1])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(2)
        check_circuit.CU3(angles, 0, 1)
        assert qickit_circuit == check_circuit

    def test_CSWAP(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(3, 3)
        cswap = SwapGate().control(1)
        qiskit_circuit.append(cswap, [0, 1, 2])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(3)
        check_circuit.CSWAP(0, 1, 2)
        assert qickit_circuit == check_circuit

    def test_MCX(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcx = XGate().control(2)
        qiskit_circuit.append(mcx, [0, 1, 2])
        qiskit_circuit.append(mcx, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCX([0, 1], 2)
        check_circuit.MCX([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCY(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcy = YGate().control(2)
        qiskit_circuit.append(mcy, [0, 1, 2])
        qiskit_circuit.append(mcy, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCY([0, 1], 2)
        check_circuit.MCY([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCZ(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcz = ZGate().control(2)
        qiskit_circuit.append(mcz, [0, 1, 2])
        qiskit_circuit.append(mcz, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCZ([0, 1], 2)
        check_circuit.MCZ([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCH(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mch = HGate().control(2)
        qiskit_circuit.append(mch, [0, 1, 2])
        qiskit_circuit.append(mch, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCH([0, 1], 2)
        check_circuit.MCH([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCS(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcs = SGate().control(2)
        qiskit_circuit.append(mcs, [0, 1, 2])
        qiskit_circuit.append(mcs, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCS([0, 1], 2)
        check_circuit.MCS([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCT(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mct = TGate().control(2)
        qiskit_circuit.append(mct, [0, 1, 2])
        qiskit_circuit.append(mct, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCT([0, 1], 2)
        check_circuit.MCT([0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRX(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcrx = RXGate(0.5).control(2)
        qiskit_circuit.append(mcrx, [0, 1, 2])
        qiskit_circuit.append(mcrx, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCRX(0.5, [0, 1], 2)
        check_circuit.MCRX(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRY(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcry = RYGate(0.5).control(2)
        qiskit_circuit.append(mcry, [0, 1, 2])
        qiskit_circuit.append(mcry, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCRY(0.5, [0, 1], 2)
        check_circuit.MCRY(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCRZ(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcrz = RZGate(0.5).control(2)
        qiskit_circuit.append(mcrz, [0, 1, 2])
        qiskit_circuit.append(mcrz, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCRZ(0.5, [0, 1], 2)
        check_circuit.MCRZ(0.5, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCU3(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        angles = [0.1, 0.2, 0.3]
        mcu3 = U3Gate(angles[0], angles[1], angles[2]).control(2)
        qiskit_circuit.append(mcu3, [0, 1, 2])
        qiskit_circuit.append(mcu3, [0, 1, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCU3(angles, [0, 1], 2)
        check_circuit.MCU3(angles, [0, 1], 3)
        assert qickit_circuit == check_circuit

    def test_MCSWAP(self) -> None:
        # Define the Qiskit circuit
        qiskit_circuit = QuantumCircuit(4, 4)
        mcswap = SwapGate().control(2)
        qiskit_circuit.append(mcswap, [0, 1, 2, 3])

        # Convert the Qiskit circuit to a QICKIT circuit
        qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

        # Define the equivalent QICKIT circuit, and ensure
        # that the two circuits are equal
        check_circuit = QiskitCircuit(4)
        check_circuit.MCSWAP([0, 1], 2, 3)
        assert qickit_circuit == check_circuit

    def test_GlobalPhase(self) -> None:
        # # Define the Qiskit circuit
        # qiskit_circuit = QuantumCircuit(1, 1)
        # qiskit_circuit.append(GlobalPhaseGate(0.5), (), ())

        # # Convert the Qiskit circuit to a QICKIT circuit
        # qickit_circuit = Circuit.from_qiskit(qiskit_circuit, QiskitCircuit)

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