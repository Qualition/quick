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

# Pennylane imports
import pennylane as qml

# QICKIT imports
from qickit.circuit import PennylaneCircuit
from tests.circuit import Template


# PennyLane's `.matrix` function does not take qubit ordering into account,
# so we need to manually convert the unitary matrix from MSB to LSB
def MSB_to_LSB(matrix: NDArray[np.number]) -> NDArray[np.number]:
    """ Convert the MSB to LSB.

    Parameters
    ----------
    `matrix` (NDArray[np.number]):
        The matrix to convert.

    Returns
    -------
    `reordered_matrix` (NDArray[np.number]): The new matrix with LSB conversion.
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
    """ `qickit.TestPennylaneCircuit` is the tester class for `qickit.PennylaneCircuit` class.
    """
    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Pauli-X gate
        circuit.X(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.PauliX(0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Pauli-Y gate
        circuit.Y(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.PauliY(0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Pauli-Z gate
        circuit.Z(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.PauliZ(0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.Hadamard(0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the S gate
        circuit.S(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.S(0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the T gate
        circuit.T(0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.T(0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.RX(0.5, 0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.RY(0.5, 0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.RZ(0.5, 0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.5, 0.5, 0.5], 0)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.U3(0.5, 0.5, 0.5, 0)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CNOT([0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CY([0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CZ([0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CH([0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        cs = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=0, wires=1)

        def pennylane_circuit() -> float:
            qml.apply(cs)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CT(self) -> None:
        """ Test the Controlled Clifford-T gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        ct = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=0, wires=1)

        def pennylane_circuit() -> float:
            qml.apply(ct)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CRX(0.5, [0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CRY(0.5, [0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        def pennylane_circuit() -> float:
            qml.CRZ(0.5, [0, 1])

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.5, 0.5, 0.5], 0, 1)

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        u3 = qml.U3(theta=0.5, phi=0.5, delta=0.5,wires=0).matrix()
        cu3 = qml.ControlledQubitUnitary(u3, control_wires=0, wires=1)

        def pennylane_circuit() -> float:
            qml.apply(cu3)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcx_1 = qml.ControlledQubitUnitary(qml.PauliX(0).matrix(), control_wires=[0, 1], wires=[2])
        mcx_2 = qml.ControlledQubitUnitary(qml.PauliX(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcx_1)
            qml.apply(mcx_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcy_1 = qml.ControlledQubitUnitary(qml.PauliY(0).matrix(), control_wires=[0, 1], wires=[2])
        mcy_2 = qml.ControlledQubitUnitary(qml.PauliY(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcy_1)
            qml.apply(mcy_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcz_1 = qml.ControlledQubitUnitary(qml.PauliZ(0).matrix(), control_wires=[0, 1], wires=[2])
        mcz_2 = qml.ControlledQubitUnitary(qml.PauliZ(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcz_1)
            qml.apply(mcz_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mch_1 = qml.ControlledQubitUnitary(qml.Hadamard(0).matrix(), control_wires=[0, 1], wires=[2])
        mch_2 = qml.ControlledQubitUnitary(qml.Hadamard(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mch_1)
            qml.apply(mch_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcs_1 = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=[0, 1], wires=[2])
        mcs_2 = qml.ControlledQubitUnitary(qml.S(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcs_1)
            qml.apply(mcs_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCT(self) -> None:
        """ Test the Multi-Controlled Clifford-T gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mct_1 = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=[0, 1], wires=[2])
        mct_2 = qml.ControlledQubitUnitary(qml.T(0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mct_1)
            qml.apply(mct_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcrx_1 = qml.ControlledQubitUnitary(qml.RX(0.5, 0).matrix(), control_wires=[0, 1], wires=[2])
        mcrx_2 = qml.ControlledQubitUnitary(qml.RX(0.5, 0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcrx_1)
            qml.apply(mcrx_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcry_1 = qml.ControlledQubitUnitary(qml.RY(0.5, 0).matrix(), control_wires=[0, 1], wires=[2])
        mcry_2 = qml.ControlledQubitUnitary(qml.RY(0.5, 0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcry_1)
            qml.apply(mcry_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        mcrz_1 = qml.ControlledQubitUnitary(qml.RZ(0.5, 0).matrix(), control_wires=[0, 1], wires=[2])
        mcrz_2 = qml.ControlledQubitUnitary(qml.RZ(0.5, 0).matrix(), control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcrz_1)
            qml.apply(mcrz_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """
        # Define the `qickit.PennylaneCircuit` instance
        circuit = PennylaneCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])

        # Since Pennylane does not have a circuit object, we will
        # define the equivalent `qml.QNode` instance, and ensure
        # the results are equivalent
        u3 = qml.U3(theta=0.5, phi=0.5, delta=0.5, wires=0).matrix()

        mcu3_1 = qml.ControlledQubitUnitary(u3, control_wires=[0, 1], wires=[2])
        mcu3_2 = qml.ControlledQubitUnitary(u3, control_wires=[0, 1], wires=[3])

        def pennylane_circuit() -> float:
            qml.apply(mcu3_1)
            qml.apply(mcu3_2)

        # Compare the results
        assert np.all(circuit.get_unitary() == MSB_to_LSB(np.array(qml.matrix(pennylane_circuit)(), dtype=complex)))

    def test_measure(self) -> None:
        return super().test_measure()