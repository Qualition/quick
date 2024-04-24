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

__all__ = ['TestPennylaneCircuit']

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal

# Pennylane imports
import pennylane as qml # type: ignore

# QICKIT imports
from qickit.circuit import PennylaneCircuit
from tests.circuit import Template


# PennyLane's `.matrix` function does not take qubit ordering into account,
# so we need to manually convert the unitary matrix from MSB to LSB
def MSB_to_LSB(matrix: NDArray[np.number]) -> NDArray[np.number]:
    """ Convert the MSB to LSB.

    Parameters
    ----------
    `matrix` : NDArray[np.number]
        The matrix to convert.

    Returns
    -------
    `reordered_matrix` : NDArray[np.number]
        The new matrix with LSB conversion.
    """
    # Determine the size of the matrix (assuming it's a square matrix)
    size = len(matrix)

    # Create a new matrix to store the reordered elements
    reordered_matrix = np.zeros((size, size), dtype=type(matrix[0][0]))

    # Iterate over each element in the original matrix
    for i in range(size):
        for j in range(size):
            # Convert the indices from MSB to LSB
            new_i = int(bin(i)[2:].zfill(int(np.log2(size)))[::-1], 2)
            new_j = int(bin(j)[2:].zfill(int(np.log2(size)))[::-1], 2)

            # Assign the value from the original matrix to the new position in the reordered matrix
            reordered_matrix[new_i][new_j] = matrix[i][j]

    # Return the reordered matrix
    return reordered_matrix

class TestPennylaneCircuit(Template):
    """ `tests.circuit.TestPennylaneCircuit` is the tester class for `qickit.circuit.PennylaneCircuit` class.
    """
    def test_X(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Pauli-X gate
        circuit.X(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.PauliX(0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_Y(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Pauli-Y gate
        circuit.Y(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.PauliY(0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_Z(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Pauli-Z gate
        circuit.Z(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.PauliZ(0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_H(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.Hadamard(0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_S(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the S gate
        circuit.S(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.S(0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_T(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the T gate
        circuit.T(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.T(0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_RX(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.RX(0.5, 0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_RY(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.RY(0.5, 0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_RZ(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.RZ(0.5, 0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_U3(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.1, 0.2, 0.3], 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.U3(theta=0.1, phi=0.2, delta=0.3, wires=0)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_SWAP(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the SWAP gate
        circuit.SWAP(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.SWAP([0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CX(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CNOT([0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CY(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CY([0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CZ(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CZ([0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CH(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CH([0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CS(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        cs = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=0, wires=1)

        def pennylane_circuit() -> None:
            qml.apply(cs)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CT(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        ct = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=0, wires=1)

        def pennylane_circuit() -> None:
            qml.apply(ct)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CRX(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CRX(0.5, [0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CRY(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CRY(0.5, [0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CRZ(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CRZ(0.5, [0, 1])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CU3(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.1, 0.2, 0.3], 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        u3 = qml.U3(theta=0.1, phi=0.2, delta=0.3,wires=0).matrix()
        cu3 = qml.ControlledQubitUnitary(u3, control_wires=0, wires=1)

        def pennylane_circuit() -> None:
            qml.apply(cu3)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_CSWAP(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(3, 3)

        # Apply the CSWAP gate
        circuit.CSWAP(0, 1, 2)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> None:
            qml.CSWAP([0, 1, 2])

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCX(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcx_1 = qml.ControlledQubitUnitary(qml.PauliX(0).matrix(), control_wires=[0, 1], wires=[2])
        mcx_2 = qml.ControlledQubitUnitary(qml.PauliX(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcx_1)
            qml.apply(mcx_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCY(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcy_1 = qml.ControlledQubitUnitary(qml.PauliY(0).matrix(), control_wires=[0, 1], wires=[2])
        mcy_2 = qml.ControlledQubitUnitary(qml.PauliY(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcy_1)
            qml.apply(mcy_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCZ(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcz_1 = qml.ControlledQubitUnitary(qml.PauliZ(0).matrix(), control_wires=[0, 1], wires=[2])
        mcz_2 = qml.ControlledQubitUnitary(qml.PauliZ(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcz_1)
            qml.apply(mcz_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCH(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mch_1 = qml.ControlledQubitUnitary(qml.Hadamard(0).matrix(), control_wires=[0, 1], wires=[2])
        mch_2 = qml.ControlledQubitUnitary(qml.Hadamard(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mch_1)
            qml.apply(mch_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCS(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcs_1 = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=[0, 1], wires=[2])
        mcs_2 = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcs_1)
            qml.apply(mcs_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCT(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mct_1 = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=[0, 1], wires=[2])
        mct_2 = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mct_1)
            qml.apply(mct_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCRX(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcrx_1 = qml.ControlledQubitUnitary(qml.RX(0.5, 0).matrix(), control_wires=[0, 1], wires=[2])
        mcrx_2 = qml.ControlledQubitUnitary(qml.RX(0.5, 0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcrx_1)
            qml.apply(mcrx_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCRY(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcry_1 = qml.ControlledQubitUnitary(qml.RY(0.5, 0).matrix(), control_wires=[0, 1], wires=[2])
        mcry_2 = qml.ControlledQubitUnitary(qml.RY(0.5, 0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcry_1)
            qml.apply(mcry_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCRZ(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcrz_1 = qml.ControlledQubitUnitary(qml.RZ(0.5, 0).matrix(), control_wires=[0, 1], wires=[2])
        mcrz_2 = qml.ControlledQubitUnitary(qml.RZ(0.5, 0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcrz_1)
            qml.apply(mcrz_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCU3(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        u3 = qml.U3(theta=0.1, phi=0.2, delta=0.3, wires=0).matrix()

        mcu3_1 = qml.ControlledQubitUnitary(u3, control_wires=[0, 1], wires=[2])
        mcu3_2 = qml.ControlledQubitUnitary(u3, control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> None:
            qml.apply(mcu3_1)
            qml.apply(mcu3_2)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_MCSWAP(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCSWAP gate
        circuit.MCSWAP([0, 1], 2, 3)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcswap_1 = qml.ControlledQubitUnitary(qml.SWAP([0, 1]).matrix(), control_wires=[0, 1], wires=[2, 3])

        def pennylane_circuit() -> None:
            qml.apply(mcswap_1)

        # Compare the results
        assert_almost_equal(circuit.get_unitary(), MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)), 8)

    def test_GlobalPhase(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)

        # Ensure the global phase is correct
        assert_almost_equal(circuit.get_unitary(), np.exp(1.8j) * np.eye(2), 8)

    def test_measure(self) -> None:
        return super().test_measure()

    def test_unitary(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the gate
        circuit.MCX([0, 1], [2, 3])

        # Define the unitary
        unitary = circuit.get_unitary()

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        unitary_circuit = PennylaneCircuit(4, 4)
        unitary_circuit.unitary(unitary, [0, 1, 2, 3])

        assert_almost_equal(unitary_circuit.get_unitary(), unitary, 8)

    def test_vertical_reverse(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply GHZ state
        circuit.H(0)
        circuit.CX(0, 1)

        # Apply the vertical reverse operation
        circuit.vertical_reverse()

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = PennylaneCircuit(2, 2)
        updated_circuit.H(1)
        updated_circuit.CX(1, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_horizontal_reverse(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply a RX and CX gate
        circuit.RX(np.pi, 0)
        circuit.CX(0, 1)

        # Apply the horizontal reverse operation
        circuit.horizontal_reverse()

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = PennylaneCircuit(2, 2)
        updated_circuit.CX(0, 1)
        updated_circuit.RX(-np.pi, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_add(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instances
        circuit1 = PennylaneCircuit(2, 2)
        circuit2 = PennylaneCircuit(2, 2)

        # Apply the Pauli-X gate
        circuit1.CX(0, 1)
        circuit2.CY(1, 0)

        # Add the two circuits
        circuit1.add(circuit2, [0, 1])

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        added_circuit = PennylaneCircuit(2, 2)
        added_circuit.CX(0, 1)
        added_circuit.CY(1, 0)

        assert circuit1 == added_circuit
        assert_almost_equal(circuit1.get_unitary(), added_circuit.get_unitary(), 8)

    def test_transpile(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        transpiled_circuit = PennylaneCircuit(4, 4)
        transpiled_circuit.MCX([0, 1], [2, 3])
        transpiled_circuit.transpile()

        assert_almost_equal(circuit.get_unitary(), transpiled_circuit.get_unitary(), 8)

    def test_get_depth(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Get the depth of the circuit, and ensure it is correct
        depth = circuit.get_depth()

        assert depth == 21

    def test_get_width(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Get the width of the circuit, and ensure it is correct
        width = circuit.get_width()

        assert width == 4

    def test_compress(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the MCX gate
        circuit.RX(np.pi/2, 0)

        # Compress the circuit
        circuit.compress(1.0)

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        compressed_circuit = PennylaneCircuit(1, 1)

        assert circuit == compressed_circuit
        assert_almost_equal(circuit.get_unitary(), compressed_circuit.get_unitary(), 8)

    def test_change_mapping(self) -> None:
        # Define the `qickit.circuit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Change the mapping of the circuit
        circuit.change_mapping([3, 2, 1, 0])

        # Define the equivalent `qickit.circuit.PennylaneCircuit` instance, and
        # ensure they are equivalent
        mapped_circuit = PennylaneCircuit(4, 4)
        mapped_circuit.MCX([3, 2], [1, 0])

        assert circuit == mapped_circuit
        assert_almost_equal(circuit.get_unitary(), mapped_circuit.get_unitary(), 8)