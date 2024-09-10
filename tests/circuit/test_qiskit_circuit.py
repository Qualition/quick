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

__all__ = ["TestQiskitCircuit"]

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from scipy.spatial import distance # type: ignore

from qiskit import QuantumCircuit

from qickit.circuit import QiskitCircuit
from tests.circuit import Template
from tests.circuit.gate_utils import (
    X_unitary_matrix, Y_unitary_matrix, Z_unitary_matrix,
    H_unitary_matrix, S_unitary_matrix, T_unitary_matrix,
    RX_unitary_matrix, RY_unitary_matrix, RZ_unitary_matrix,
    U3_unitary_matrix, SWAP_unitary_matrix,
    CX_unitary_matrix, CY_unitary_matrix, CZ_unitary_matrix,
    CH_unitary_matrix, CS_unitary_matrix, CT_unitary_matrix,
    CRX_unitary_matrix, CRY_unitary_matrix, CRZ_unitary_matrix,
    CU3_unitary_matrix, CSWAP_unitary_matrix,
    MCX_unitary_matrix, MCY_unitary_matrix, MCZ_unitary_matrix,
    MCH_unitary_matrix, MCS_unitary_matrix, MCT_unitary_matrix,
    MCRX_unitary_matrix, MCRY_unitary_matrix, MCRZ_unitary_matrix,
    MCU3_unitary_matrix, MCSWAP_unitary_matrix, Identity_unitary_matrix
)


def cosine_similarity(
        h1: dict[str, int],
        h2: dict[str, int]
    ) -> float:
    """ Calculate the cosine similarity between two histograms.

    Parameters
    ----------
    h1 : dict[str, int]
        The first histogram.
    h2 : dict[str, int]
        The second histogram.

    Returns
    -------
    float
        The cosine similarity between the two histograms.
    """
    # Convert dictionaries to lists
    keys = set(h1.keys()).union(h2.keys())
    dist_1 = [h1.get(key, 0) for key in keys]
    dist_2 = [h2.get(key, 0) for key in keys]

    return float(1 - distance.cosine(dist_1, dist_2))


class TestQiskitCircuit(Template):
    """ `tests.circuit.TestQiskitCircuit` is the tester class for `qickit.circuit.QiskitCircuit` class.
    """
    def test_init(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

    def test_num_qubits_value(self) -> None:
        # Ensure the error is raised when the number of qubits is less than or equal to 0
        with pytest.raises(ValueError):
            circuit = QiskitCircuit(0)

        with pytest.raises(ValueError):
            circuit = QiskitCircuit(-1)

    def test_num_qubits_type(self) -> None:
        # Ensure the error is raised when the number of qubits is not an integer
        with pytest.raises(TypeError):
            circuit = QiskitCircuit(1.0) # type: ignore

    def test_single_qubit_gate_from_range(self) -> None:
        """ Test the single qubit gate when indices are passed as a range instance.
        """
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(3)

        # Define the qubit indices as a range of ints
        qubit_indices = range(3)

        # Apply the Pauli-X gate
        circuit.X(qubit_indices)

        # Define the checker
        checker_circuit = QiskitCircuit(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    def test_single_qubit_gate_from_tuple(self) -> None:
        """ Test the single qubit gate when indices are passed as a tuple instance.
        """
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(3)

        # Define the qubit indices as a tuple of ints
        qubit_indices = (0, 1, 2)

        # Apply the Pauli-X gate
        circuit.X(qubit_indices)

        # Define the checker
        checker_circuit = QiskitCircuit(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    def test_single_qubit_gate_from_ndarray(self) -> None:
        """ Test the single qubit gate when indices are passed as a numpy.ndarray instance.
        """
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(3)

        # Define the qubit indices as a ndarray of ints
        qubit_indices = np.array([0, 1, 2])

        # Apply the Pauli-X gate
        circuit.X(qubit_indices) # type: ignore

        # Define the checker
        checker_circuit = QiskitCircuit(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    def test_Identity(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the Identity gate
        circuit.Identity(0)

        assert_almost_equal(circuit.get_unitary(), Identity_unitary_matrix, 8)

    def test_X(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the Pauli X gate
        circuit.X(0)

        assert_almost_equal(circuit.get_unitary(), X_unitary_matrix, 8)

    def test_Y(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the Pauli Y gate
        circuit.Y(0)

        assert_almost_equal(circuit.get_unitary(), Y_unitary_matrix, 8)

    def test_Z(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the Pauli Z gate
        circuit.Z(0)

        assert_almost_equal(circuit.get_unitary(), Z_unitary_matrix, 8)

    def test_H(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the Hadamard gate
        circuit.H(0)

        assert_almost_equal(circuit.get_unitary(), H_unitary_matrix, 8)

    def test_S(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the S gate
        circuit.S(0)

        assert_almost_equal(circuit.get_unitary(), S_unitary_matrix, 8)

    def test_Sdg(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the S dagger gate
        circuit.Sdg(0)

        assert_almost_equal(circuit.get_unitary(), S_unitary_matrix.conj().T, 8)

    def test_T(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the T gate
        circuit.T(0)

        assert_almost_equal(circuit.get_unitary(), T_unitary_matrix, 8)

    def test_Tdg(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the T dagger gate
        circuit.Tdg(0)

        assert_almost_equal(circuit.get_unitary(), T_unitary_matrix.conj().T, 8)

    def test_RX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the RX gate
        circuit.RX(np.pi/4, 0)

        assert_almost_equal(circuit.get_unitary(), RX_unitary_matrix, 8)

    def test_RY(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the RY gate
        circuit.RY(np.pi/4, 0)

        assert_almost_equal(circuit.get_unitary(), RY_unitary_matrix, 8)

    def test_RZ(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the RZ gate
        circuit.RZ(np.pi/4, 0)

        assert_almost_equal(circuit.get_unitary(), RZ_unitary_matrix, 8)

    def test_U3(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the U3 gate
        circuit.U3([np.pi/2, np.pi/3, np.pi/4], 0)

        assert_almost_equal(circuit.get_unitary(), U3_unitary_matrix, 8)

    def test_SWAP(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the SWAP gate
        circuit.SWAP(0, 1)

        assert_almost_equal(circuit.get_unitary(), SWAP_unitary_matrix, 8)

    def test_CX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CX gate
        circuit.CX(0, 1)

        assert_almost_equal(circuit.get_unitary(), CX_unitary_matrix, 8)

    def test_CY(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CY gate
        circuit.CY(0, 1)

        assert_almost_equal(circuit.get_unitary(), CY_unitary_matrix, 8)

    def test_CZ(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        assert_almost_equal(circuit.get_unitary(), CZ_unitary_matrix, 8)

    def test_CH(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CH gate
        circuit.CH(0, 1)

        assert_almost_equal(circuit.get_unitary(), CH_unitary_matrix, 8)

    def test_CS(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CS gate
        circuit.CS(0, 1)

        assert_almost_equal(circuit.get_unitary(), CS_unitary_matrix, 8)

    def test_CSdg(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CSdg gate
        circuit.CSdg(0, 1)

        assert_almost_equal(circuit.get_unitary(), CS_unitary_matrix.conj().T, 8)

    def test_CT(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CT gate
        circuit.CT(0, 1)

        assert_almost_equal(circuit.get_unitary(), CT_unitary_matrix, 8)

    def test_CTdg(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CTdg gate
        circuit.CTdg(0, 1)

        assert_almost_equal(circuit.get_unitary(), CT_unitary_matrix.conj().T, 8)

    def test_CRX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CRX gate
        circuit.CRX(np.pi/4, 0, 1)

        assert_almost_equal(circuit.get_unitary(), CRX_unitary_matrix, 8)

    def test_CRY(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CRY gate
        circuit.CRY(np.pi/4, 0, 1)

        assert_almost_equal(circuit.get_unitary(), CRY_unitary_matrix, 8)

    def test_CRZ(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CRZ gate
        circuit.CRZ(np.pi/4, 0, 1)

        assert_almost_equal(circuit.get_unitary(), CRZ_unitary_matrix, 8)

    def test_CU3(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the CU3 gate
        circuit.CU3([np.pi/2, np.pi/3, np.pi/4], 0, 1)

        assert_almost_equal(circuit.get_unitary(), CU3_unitary_matrix, 8)

    def test_CSWAP(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(3)

        # Apply the CSWAP gate
        circuit.CSWAP(0, 1, 2)

        assert_almost_equal(circuit.get_unitary(), CSWAP_unitary_matrix, 8)

    def test_MCX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCX_unitary_matrix, 8)

    def test_MCY(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCY_unitary_matrix, 8)

    def test_MCZ(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCZ_unitary_matrix, 8)

    def test_MCH(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCH_unitary_matrix, 8)

    def test_MCS(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCS_unitary_matrix, 8)

    def test_MCSdg(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCSdg gate
        circuit.MCSdg([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCS_unitary_matrix.conj().T, 8)

    def test_MCT(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCT_unitary_matrix, 8)

    def test_MCTdg(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCTdg gate
        circuit.MCTdg([0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCT_unitary_matrix.conj().T, 8)

    def test_MCRX(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCRX gate
        circuit.MCRX(np.pi/4, [0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCRX_unitary_matrix, 8)

    def test_MCRY(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCRY gate
        circuit.MCRY(np.pi/4, [0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCRY_unitary_matrix, 8)

    def test_MCRZ(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCRZ gate
        circuit.MCRZ(np.pi/4, [0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCRZ_unitary_matrix, 8)

    def test_MCU3(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCU3 gate
        circuit.MCU3([np.pi/2, np.pi/3, np.pi/4], [0, 1], [2, 3])

        assert_almost_equal(circuit.get_unitary(), MCU3_unitary_matrix, 8)

    def test_MCSWAP(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCSWAP gate
        circuit.MCSWAP([0, 1], 2, 3)

        assert_almost_equal(circuit.get_unitary(), MCSWAP_unitary_matrix, 8)

    def test_GlobalPhase(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the global phase gate
        circuit.GlobalPhase(1.8)

        # Ensure the global phase is correct
        assert_almost_equal(circuit.get_unitary(), np.exp(1.8j) * np.eye(2), 8)

    def test_single_measurement(self) -> None:
        # Define `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Ensure no qubits are measured initially
        assert len(circuit.measured_qubits) == 0

        # Apply the measurement gate
        circuit.measure(0)

        # Ensure only the first qubit is measured
        assert circuit.measured_qubits == {0}

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(1, 1)
        qiskit_circuit.measure(0, 0)

        assert [data.operation for data in circuit.circuit] == [data.operation for data in qiskit_circuit]

    def test_multiple_measurement(self) -> None:
        # Define `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Ensure no qubits are measured initially
        assert len(circuit.measured_qubits) == 0

        # Apply the measurement gate
        circuit.measure([0, 1])

        # Ensure both qubits are measured
        assert circuit.measured_qubits == {0, 1}

        # Define the equivalent `qiskit.QuantumCircuit` instance, and
        # ensure they are equivalent
        qiskit_circuit = QuantumCircuit(2, 2)
        qiskit_circuit.measure(0, 0)
        qiskit_circuit.measure(1, 1)

        assert [data.operation for data in circuit.circuit] == [data.operation for data in qiskit_circuit]

    def test_remove_measurement(self) -> None:
        # Define the `qickit.circuit.CirqCircuit` instance
        circuit = QiskitCircuit(2)
        no_measurement_circuit = QiskitCircuit(2)

        # Apply the measurement gate
        circuit.measure([0, 1])

        # Ensure both qubits are measured
        assert circuit.measured_qubits == {0, 1}

        # Remove the measurement gate
        circuit = circuit._remove_measurements()

        # Ensure no qubits are measured
        assert len(circuit.measured_qubits) == 0

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        assert circuit == no_measurement_circuit

    def test_unitary(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the gate
        circuit.MCX([0, 1], [2, 3])

        # Define the unitary
        unitary = circuit.get_unitary()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        unitary_circuit = QiskitCircuit(4)
        unitary_circuit.unitary(unitary, [0, 1, 2, 3])

        assert_almost_equal(unitary_circuit.get_unitary(), unitary, 8)

    def test_get_statevector(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Create a state with non-zero real and imaginary components
        circuit.H(0)
        circuit.T(0)

        # Get the statevector of the circuit, and ensure it is correct
        statevector = circuit.get_statevector()

        assert_almost_equal(statevector, [np.sqrt(1/2), 0.5 + 0.5j], 8)

    def test_partial_get_counts(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(3)

        # Prepare the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Perform partial measurement on the first qubit and ensure the counts are correct
        circuit.measure(0)
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {"0": 512, "1": 512}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the second qubit and ensure the counts are correct
        circuit.measure(1)
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {"0": 512, "1": 512}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the third qubit and ensure the counts are correct
        circuit.measure(2)
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {"0": 1024, "1": 0}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the first and second qubits and ensure the counts are correct
        circuit.measure([0, 1])
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {'00': 512, '01': 0, '10': 0, '11': 512}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the first and third qubits and ensure the counts are correct
        circuit.measure([0, 2])
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {'00': 512, '01': 512, '10': 0, '11': 0}) > 0.95

    def test_get_counts(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Measure the circuit
        circuit.measure_all()

        # Get the counts of the circuit, and ensure it is correct
        counts = circuit.get_counts(1024)

        assert cosine_similarity(counts, {"00": 512, "01": 0, "10":0, "11": 512}) > 0.95

    def test_vertical_reverse(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply GHZ state
        circuit.H(0)
        circuit.CX(0, 1)

        # Apply the vertical reverse operation
        circuit.vertical_reverse()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = QiskitCircuit(2)
        updated_circuit.H(1)
        updated_circuit.CX(1, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_horizontal_reverse(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply a gates
        circuit.RX(np.pi, 0)
        circuit.CRX(np.pi, 0, 1)
        circuit.MCRX(np.pi, [0, 1], [2, 3])
        circuit.RY(np.pi, 0)
        circuit.CRY(np.pi, 0, 1)
        circuit.MCRY(np.pi, [0, 1], [2, 3])
        circuit.RZ(np.pi, 0)
        circuit.CRZ(np.pi, 0, 1)
        circuit.MCRZ(np.pi, [0, 1], [2, 3])
        circuit.S(0)
        circuit.T(0)
        circuit.CS(0, 1)
        circuit.CT(0, 1)
        circuit.MCS([0, 1], [2, 3])
        circuit.MCT([0, 1], [2, 3])
        circuit.U3([np.pi/2, np.pi/3, np.pi/4], 0)
        circuit.CU3([np.pi/2, np.pi/3, np.pi/4], 0, 1)
        circuit.MCU3([np.pi/2, np.pi/3, np.pi/4], [0, 1], [2, 3])
        circuit.CX(0, 1)

        # Apply the horizontal reverse operation
        circuit.horizontal_reverse()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        updated_circuit = QiskitCircuit(4)
        updated_circuit.CX(0, 1)
        updated_circuit.MCU3([-np.pi/2, -np.pi/4, -np.pi/3], [0, 1], [2, 3])
        updated_circuit.CU3([-np.pi/2, -np.pi/4, -np.pi/3], 0, 1)
        updated_circuit.U3([-np.pi/2, -np.pi/4, -np.pi/3], 0)
        updated_circuit.MCTdg([0, 1], [2, 3])
        updated_circuit.MCSdg([0, 1], [2, 3])
        updated_circuit.CTdg(0, 1)
        updated_circuit.CSdg(0, 1)
        updated_circuit.Tdg(0)
        updated_circuit.Sdg(0)
        updated_circuit.MCRZ(-np.pi, [0, 1], [2, 3])
        updated_circuit.CRZ(-np.pi, 0, 1)
        updated_circuit.RZ(-np.pi, 0)
        updated_circuit.MCRY(-np.pi, [0, 1], [2, 3])
        updated_circuit.CRY(-np.pi, 0, 1)
        updated_circuit.RY(-np.pi, 0)
        updated_circuit.MCRX(-np.pi, [0, 1], [2, 3])
        updated_circuit.CRX(-np.pi, 0, 1)
        updated_circuit.RX(-np.pi, 0)

        assert circuit == updated_circuit
        assert_almost_equal(circuit.get_unitary(), updated_circuit.get_unitary(), 8)

    def test_add(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instances
        circuit1 = QiskitCircuit(2)
        circuit2 = QiskitCircuit(2)

        # Apply the gates
        circuit1.CX(0, 1)
        circuit2.CY(0, 1)
        circuit2.H(0)

        # Add the two circuits
        circuit1.add(circuit2, [1, 0])

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        added_circuit = QiskitCircuit(2)
        added_circuit.CX(0, 1)
        added_circuit.CY(1, 0)
        added_circuit.H(1)

        assert circuit1 == added_circuit
        assert_almost_equal(circuit1.get_unitary(), added_circuit.get_unitary(), 8)

    def test_add_fail(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instances
        circuit1 = QiskitCircuit(2)
        circuit2 = QiskitCircuit(3)
        circuit3 = "circuit"

        # Ensure the error is raised when the type of the circuit is not correct
        with pytest.raises(TypeError):
            circuit1.add(circuit3, [0, 1]) # type: ignore

        # Ensure the error is raised when the number of qubits is not equal
        with pytest.raises(ValueError):
            circuit1.add(circuit2, [0, 1])

    def test_transpile(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        transpiled_circuit = QiskitCircuit(4)
        transpiled_circuit.MCX([0, 1], [2, 3])
        transpiled_circuit.transpile()

        assert_almost_equal(circuit.get_unitary(), transpiled_circuit.get_unitary(), 8)

    def test_get_depth(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Get the depth of the circuit, and ensure it is correct
        depth = circuit.get_depth()

        assert depth == 21

    def test_get_width(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Get the width of the circuit, and ensure it is correct
        width = circuit.get_width()

        assert width == 4

    def test_compress(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Apply the MCX gate
        circuit.RX(np.pi/2, 0)

        # Compress the circuit
        circuit.compress(1.0)

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        compressed_circuit = QiskitCircuit(1)

        assert circuit == compressed_circuit
        assert_almost_equal(circuit.get_unitary(), compressed_circuit.get_unitary(), 8)

    def test_compress_fail(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(1)

        # Ensure the error is raised when the compression factor is less than 0
        with pytest.raises(ValueError):
            circuit.compress(-1.0)

        # Ensure the error is raised when the compression factor is greater than 1
        with pytest.raises(ValueError):
            circuit.compress(2.0)

    def test_change_mapping(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Change the mapping of the circuit
        circuit.change_mapping([3, 2, 1, 0])

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        mapped_circuit = QiskitCircuit(4)
        mapped_circuit.MCX([3, 2], [1, 0])

        assert circuit == mapped_circuit
        assert_almost_equal(circuit.get_unitary(), mapped_circuit.get_unitary(), 8)

    def test_reset(self) -> None:
        # Define the `qickit.circuit.QiskitCircuit` instance
        circuit = QiskitCircuit(2)

        # Apply the Hadamard gate
        circuit.H(0)

        # Reset the circuit
        circuit.reset()

        # Define the equivalent `qickit.circuit.QiskitCircuit` instance, and
        # ensure they are equivalent
        reset_circuit = QiskitCircuit(2)

        assert circuit == reset_circuit