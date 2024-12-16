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

__all__ = ["TestAerBackend"]

import numpy as np # type: ignore
from numpy.typing import NDArray
from scipy.spatial import distance # type: ignore

from qickit.backend import AerBackend
from qickit.circuit import Circuit, CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit
from tests.backend import Template


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


class TestAerBackend(Template):
    """ `TestAerBackend` is the tester for the `AerBackend` class.
    """
    def test_init(self) -> None:
        AerBackend()

    def test_get_partial_counts(self) -> None:
        # Define the `qickit.backend.AerBackend` instance
        backend = AerBackend()

        # Define the `qickit.circuit.Circuit` instances
        cirq_circuit = CirqCircuit(3)
        pennylane_circuit = PennylaneCircuit(3)
        qiskit_circuit = QiskitCircuit(3)
        tket_circuit = TKETCircuit(3)

        # Prepare the Bell state
        cirq_circuit.H(0)
        cirq_circuit.CX(0, 1)

        pennylane_circuit.H(0)
        pennylane_circuit.CX(0, 1)

        qiskit_circuit.H(0)
        qiskit_circuit.CX(0, 1)

        tket_circuit.H(0)
        tket_circuit.CX(0, 1)

        def test_partial_measurement(circuit: Circuit) -> None:
            """ Define the circuits for partial measurement.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                The circuit to perform partial measurement on.
            """
            # Perform partial measurement on the first qubit and ensure the counts are correct
            circuit.measure(0)
            counts = backend.get_counts(circuit=circuit, num_shots=1024)
            assert cosine_similarity(counts, {"0": 512, "1": 512}) > 0.95

            circuit = circuit._remove_measurements()

            # Perform partial measurement on the second qubit and ensure the counts are correct
            circuit.measure(1)
            counts = backend.get_counts(circuit=circuit, num_shots=1024)
            assert cosine_similarity(counts, {"0": 512, "1": 512}) > 0.95

            circuit = circuit._remove_measurements()

            # Perform partial measurement on the third qubit and ensure the counts are correct
            circuit.measure(2)
            counts = backend.get_counts(circuit=circuit, num_shots=1024)
            assert cosine_similarity(counts, {"0": 1024, "1": 0}) > 0.95

            circuit = circuit._remove_measurements()

            # Perform partial measurement on the first and second qubits and ensure the counts are correct
            circuit.measure([0, 1])
            counts = backend.get_counts(circuit=circuit, num_shots=1024)
            assert cosine_similarity(counts, {'00': 512, '01': 0, '10': 0, '11': 512}) > 0.95

            circuit = circuit._remove_measurements()

            # Perform partial measurement on the first and third qubits and ensure the counts are correct
            circuit.measure([0, 2])
            counts = backend.get_counts(circuit=circuit, num_shots=1024)
            assert cosine_similarity(counts, {'00': 512, '01': 512, '10': 0, '11': 0}) > 0.95

        test_partial_measurement(cirq_circuit)
        test_partial_measurement(pennylane_circuit)
        test_partial_measurement(qiskit_circuit)
        test_partial_measurement(tket_circuit)

    def test_get_counts(self) -> None:
        # Define the `qickit.backend.AerBackend` instance
        backend = AerBackend()

        # Define the `qickit.circuit.Circuit` instances
        cirq_circuit = CirqCircuit(2)
        pennylane_circuit = PennylaneCircuit(2)
        qiskit_circuit = QiskitCircuit(2)
        tket_circuit = TKETCircuit(2)

        # Prepare the Bell state
        cirq_circuit.H(0)
        cirq_circuit.CX(0, 1)

        pennylane_circuit.H(0)
        pennylane_circuit.CX(0, 1)

        qiskit_circuit.H(0)
        qiskit_circuit.CX(0, 1)

        tket_circuit.H(0)
        tket_circuit.CX(0, 1)

        # Measure the circuits
        cirq_circuit.measure_all()
        pennylane_circuit.measure_all()
        qiskit_circuit.measure_all()
        tket_circuit.measure_all()

        # Define the number of shots
        num_shots = 1000

        # Get the counts of the circuit
        cirq_counts = backend.get_counts(cirq_circuit, num_shots=num_shots)
        pennylane_counts = backend.get_counts(pennylane_circuit, num_shots=num_shots)
        qiskit_counts = backend.get_counts(qiskit_circuit, num_shots=num_shots)
        tket_counts = backend.get_counts(tket_circuit, num_shots=num_shots)

        # Define the output counts for checking purposes
        output_counts = {'00': 500, '11': 500}

        # Ensure the resulting distributions are close enough (95 percent fidelity)
        assert cosine_similarity(cirq_counts, output_counts) > 0.95
        assert cosine_similarity(pennylane_counts, output_counts) > 0.95
        assert cosine_similarity(qiskit_counts, output_counts) > 0.95
        assert cosine_similarity(tket_counts, output_counts) > 0.95

    def test_get_statevector(self) -> None:
        # Define the `qickit.backend.AerBackend` instance
        backend = AerBackend()

        # Define the `qickit.circuit.Circuit` instances
        cirq_circuit = CirqCircuit(2)
        pennylane_circuit = PennylaneCircuit(2)
        qiskit_circuit = QiskitCircuit(2)
        tket_circuit = TKETCircuit(2)

        # Prepare the Bell state
        cirq_circuit.H(0)
        cirq_circuit.CX(0, 1)

        pennylane_circuit.H(0)
        pennylane_circuit.CX(0, 1)

        qiskit_circuit.H(0)
        qiskit_circuit.CX(0, 1)

        tket_circuit.H(0)
        tket_circuit.CX(0, 1)

        # Measure the circuits
        cirq_circuit.measure_all()
        pennylane_circuit.measure_all()
        qiskit_circuit.measure_all()
        tket_circuit.measure_all()

        # Get the statevector of the circuits
        cirq_statevector = backend.get_statevector(cirq_circuit)
        pennylane_statevector = backend.get_statevector(pennylane_circuit)
        qiskit_statevector = backend.get_statevector(qiskit_circuit)
        tket_statevector = backend.get_statevector(tket_circuit)

        # Define the output statevector for checking purposes
        output_statevector = [np.sqrt(1/2), 0, 0, np.sqrt(1/2)]

        # Ensure the resulting statevectors are close enough (99 percent fidelity)
        assert 1 - distance.cosine(cirq_statevector, output_statevector) > 0.99
        assert 1 - distance.cosine(pennylane_statevector, output_statevector) > 0.99
        assert 1 - distance.cosine(qiskit_statevector, output_statevector) > 0.99
        assert 1 - distance.cosine(tket_statevector, output_statevector) > 0.99

    def test_get_large_statevector(self) -> None:
        # Define the `qickit.backend.AerBackend` instance
        backend = AerBackend()

        # Define the `qickit.circuit.Circuit` instances
        cirq_circuit = CirqCircuit(10)
        pennylane_circuit = PennylaneCircuit(10)
        qiskit_circuit = QiskitCircuit(10)
        tket_circuit = TKETCircuit(10)

        # Prepare the Bell state
        cirq_circuit.H(0)
        cirq_circuit.CX(0, 1)
        cirq_circuit.CX(1, 2)
        cirq_circuit.CX(2, 3)
        cirq_circuit.CX(3, 4)
        cirq_circuit.CX(4, 5)
        cirq_circuit.CX(5, 6)
        cirq_circuit.CX(6, 7)
        cirq_circuit.CX(7, 8)
        cirq_circuit.CX(8, 9)

        pennylane_circuit.H(0)
        pennylane_circuit.CX(0, 1)
        pennylane_circuit.CX(1, 2)
        pennylane_circuit.CX(2, 3)
        pennylane_circuit.CX(3, 4)
        pennylane_circuit.CX(4, 5)
        pennylane_circuit.CX(5, 6)
        pennylane_circuit.CX(6, 7)
        pennylane_circuit.CX(7, 8)
        pennylane_circuit.CX(8, 9)

        qiskit_circuit.H(0)
        qiskit_circuit.CX(0, 1)
        qiskit_circuit.CX(1, 2)
        qiskit_circuit.CX(2, 3)
        qiskit_circuit.CX(3, 4)
        qiskit_circuit.CX(4, 5)
        qiskit_circuit.CX(5, 6)
        qiskit_circuit.CX(6, 7)
        qiskit_circuit.CX(7, 8)
        qiskit_circuit.CX(8, 9)

        tket_circuit.H(0)
        tket_circuit.CX(0, 1)
        tket_circuit.CX(1, 2)
        tket_circuit.CX(2, 3)
        tket_circuit.CX(3, 4)
        tket_circuit.CX(4, 5)
        tket_circuit.CX(5, 6)
        tket_circuit.CX(6, 7)
        tket_circuit.CX(7, 8)
        tket_circuit.CX(8, 9)

        # Get the statevector of the circuits
        cirq_statevector = backend.get_statevector(cirq_circuit)
        pennylane_statevector = backend.get_statevector(pennylane_circuit)
        qiskit_statevector = backend.get_statevector(qiskit_circuit)
        tket_statevector = backend.get_statevector(tket_circuit)

        # Define the output statevector for checking purposes
        output_statevector = np.zeros(2 ** 10, dtype=complex)
        output_statevector[0] = np.sqrt(1/2)
        output_statevector[1023] = np.sqrt(1/2)

        # Ensure the resulting statevectors are close enough (99 percent fidelity)
        assert 1 - distance.cosine(cirq_statevector, output_statevector) > 0.99
        assert 1 - distance.cosine(pennylane_statevector, output_statevector) > 0.99
        assert 1 - distance.cosine(qiskit_statevector, output_statevector) > 0.99
        assert 1 - distance.cosine(tket_statevector, output_statevector) > 0.99

    def test_get_unitary(self) -> None:
        # Define the `qickit.backend.AerBackend` instance
        backend = AerBackend()

        # Define the `qickit.circuit.Circuit` instances
        cirq_circuit = CirqCircuit(2)
        pennylane_circuit = PennylaneCircuit(2)
        qiskit_circuit = QiskitCircuit(2)
        tket_circuit = TKETCircuit(2)

        # Prepare the Bell state
        cirq_circuit.H(0)
        cirq_circuit.CX(0, 1)

        pennylane_circuit.H(0)
        pennylane_circuit.CX(0, 1)

        qiskit_circuit.H(0)
        qiskit_circuit.CX(0, 1)

        tket_circuit.H(0)
        tket_circuit.CX(0, 1)

        # Measure the circuits
        cirq_circuit.measure_all()
        pennylane_circuit.measure_all()
        qiskit_circuit.measure_all()
        tket_circuit.measure_all()

        # Get the unitary operator of the circuits
        cirq_operator = backend.get_operator(cirq_circuit)
        pennylane_operator = backend.get_operator(pennylane_circuit)
        qiskit_operator = backend.get_operator(qiskit_circuit)
        tket_operator = backend.get_operator(tket_circuit)

        # Define the output operator for checking purposes
        output_operator = np.array([
            [0.70710678+0.j, 0.70710678+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.70710678+0.j, -0.70710678+0.j],
            [0.+0.j, 0.+0.j, 0.70710678+0.j, 0.70710678+0.j],
            [0.70710678+0.j, -0.70710678+0.j, 0.+0.j, 0.+0.j]
        ])

        # Ensure the resulting statevectors are close enough (99 percent fidelity)
        assert 1 - distance.cosine(cirq_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(pennylane_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(qiskit_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(tket_operator.flatten(), output_operator.flatten()) > 0.99

    def test_get_large_unitary(self) -> None:
        # Define the `qickit.backend.AerBackend` instance
        backend = AerBackend()

        # Define the `qickit.circuit.Circuit` instances
        cirq_circuit = CirqCircuit(10)
        pennylane_circuit = PennylaneCircuit(10)
        qiskit_circuit = QiskitCircuit(10)
        tket_circuit = TKETCircuit(10)

        # Prepare the Bell state
        cirq_circuit.H(0)
        cirq_circuit.CX(0, 1)
        cirq_circuit.CX(1, 2)
        cirq_circuit.CX(2, 3)
        cirq_circuit.CX(3, 4)
        cirq_circuit.CX(4, 5)
        cirq_circuit.CX(5, 6)
        cirq_circuit.CX(6, 7)
        cirq_circuit.CX(7, 8)
        cirq_circuit.CX(8, 9)

        pennylane_circuit.H(0)
        pennylane_circuit.CX(0, 1)
        pennylane_circuit.CX(1, 2)
        pennylane_circuit.CX(2, 3)
        pennylane_circuit.CX(3, 4)
        pennylane_circuit.CX(4, 5)
        pennylane_circuit.CX(5, 6)
        pennylane_circuit.CX(6, 7)
        pennylane_circuit.CX(7, 8)
        pennylane_circuit.CX(8, 9)

        qiskit_circuit.H(0)
        qiskit_circuit.CX(0, 1)
        qiskit_circuit.CX(1, 2)
        qiskit_circuit.CX(2, 3)
        qiskit_circuit.CX(3, 4)
        qiskit_circuit.CX(4, 5)
        qiskit_circuit.CX(5, 6)
        qiskit_circuit.CX(6, 7)
        qiskit_circuit.CX(7, 8)
        qiskit_circuit.CX(8, 9)

        tket_circuit.H(0)
        tket_circuit.CX(0, 1)
        tket_circuit.CX(1, 2)
        tket_circuit.CX(2, 3)
        tket_circuit.CX(3, 4)
        tket_circuit.CX(4, 5)
        tket_circuit.CX(5, 6)
        tket_circuit.CX(6, 7)
        tket_circuit.CX(7, 8)
        tket_circuit.CX(8, 9)

        # Get the unitary operator of the circuits
        cirq_operator = backend.get_operator(cirq_circuit)
        pennylane_operator = backend.get_operator(pennylane_circuit)
        qiskit_operator = backend.get_operator(qiskit_circuit)
        tket_operator = backend.get_operator(tket_circuit)

        # Define the output operator for checking purposes
        output_operator: NDArray[np.complex128] = np.load("tests/backend/qiskit_backends/aer_backend_large_unitary_checker.npy")

        # Ensure the resulting operators are close enough (99 percent fidelity)
        assert 1 - distance.cosine(cirq_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(pennylane_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(qiskit_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(tket_operator.flatten(), output_operator.flatten()) > 0.99