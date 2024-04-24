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
from qiskit import QuantumCircuit # type: ignore
from qiskit.circuit.library import (RXGate, RYGate, RZGate, HGate, XGate, YGate, # type: ignore
                                    ZGate, SGate, TGate, U3Gate, SwapGate) # type: ignore
from qiskit.quantum_info import Operator # type: ignore

# QICKIT imports
from qickit.circuit import QiskitCircuit
from tests.circuit import Template


class TestQiskitCircuit(Template):
    """ `tests.circuit.TestQiskitCircuit` is the tester class for `qickit.circuit.QiskitCircuit` class.
    """
    def test_X(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.1, 0.2, 0.3], 0)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.append(U3Gate(theta=0.1, phi=0.2, lam=0.3), [0])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_SWAP(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the SWAP gate
        circuit.SWAP(0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.swap(0, 1)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.1, 0.2, 0.3], 0, 1)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        cu3 = U3Gate(theta=0.1, phi=0.2, lam=0.3).control(1)
        qiskit_circuit.append(cu3, [0, 1])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_CSWAP(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(3, 3)

        # Apply the CSWAP gate
        circuit.CSWAP(0, 1, 2)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(3, 3)
        qiskit_circuit.cswap(0, 1, 2)

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
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
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcu3 = U3Gate(theta=0.1, phi=0.2, lam=0.3).control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcu3, [0, 1, 2])
        qiskit_circuit.append(mcu3, [0, 1, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_MCSWAP(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCSWAP gate
        circuit.MCSWAP([0, 1], 2, 3)

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        mcswap = SwapGate().control(2)
        qiskit_circuit = QuantumCircuit(4, 4)
        qiskit_circuit.append(mcswap, [0, 1, 2, 3])

        result = Operator(qiskit_circuit).data

        assert_almost_equal(np.array(circuit.get_unitary()), np.array(result), 8)

    def test_GlobalPhase(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)

        # Ensure the global phase is correct
        np.isclose(circuit.circuit.global_phase, 1.8, atol=1e-8)

    def test_measure(self) -> None:
        return super().test_measure()

    def test_unitary(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the gate
        circuit.MCX([0, 1], [2, 3])

        # Define the unitary
        unitary = circuit.get_unitary()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        unitary_circuit = QiskitCircuit(4, 4)
        unitary_circuit.unitary(unitary, [0, 1, 2, 3])

        assert_almost_equal(unitary_circuit.get_unitary(), unitary, 8)

    def test_vertical_reverse(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply GHZ state
        circuit.H(0)
        circuit.CX(0, 1)

        # Apply the vertical reverse operation
        circuit.vertical_reverse()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = QiskitCircuit(2, 2)
        updated_circuit.H(1)
        updated_circuit.CX(1, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_horizontal_reverse(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2, 2)

        # Apply a RX and CX gate
        circuit.RX(np.pi, 0)
        circuit.CX(0, 1)

        # Apply the horizontal reverse operation
        circuit.horizontal_reverse()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = QiskitCircuit(2, 2)
        updated_circuit.CX(0, 1)
        updated_circuit.RX(-np.pi, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_add(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instances
        circuit1 = QiskitCircuit(2, 2)
        circuit2 = QiskitCircuit(2, 2)

        # Apply the Pauli-X gate
        circuit1.CX(0, 1)
        circuit2.CY(1, 0)

        # Add the two circuits
        circuit1.add(circuit2, [0, 1])

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        added_circuit = QiskitCircuit(2, 2)
        added_circuit.CX(0, 1)
        added_circuit.CY(1, 0)

        assert circuit1 == added_circuit
        assert_almost_equal(circuit1.get_unitary(), added_circuit.get_unitary(), 8)

    def test_transpile(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        transpiled_circuit = QiskitCircuit(4, 4)
        transpiled_circuit.MCX([0, 1], [2, 3])
        transpiled_circuit.transpile()

        assert_almost_equal(circuit.get_unitary(), transpiled_circuit.get_unitary(), 8)

    def test_get_depth(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Get the depth of the circuit, and ensure it is correct
        depth = circuit.get_depth()

        assert depth == 21

    def test_get_width(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Get the width of the circuit, and ensure it is correct
        width = circuit.get_width()

        assert width == 4

    def test_compress(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1, 1)

        # Apply the MCX gate
        circuit.RX(np.pi/2, 0)

        # Compress the circuit
        circuit.compress(1.0)

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        compressed_circuit = QiskitCircuit(1, 1)

        assert circuit == compressed_circuit
        assert_almost_equal(circuit.get_unitary(), compressed_circuit.get_unitary(), 8)

    def test_change_mapping(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Change the mapping of the circuit
        circuit.change_mapping([3, 2, 1, 0])

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        mapped_circuit = QiskitCircuit(4, 4)
        mapped_circuit.MCX([3, 2], [1, 0])

        assert circuit == mapped_circuit
        assert_almost_equal(circuit.get_unitary(), mapped_circuit.get_unitary(), 8)