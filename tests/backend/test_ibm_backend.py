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

__all__ = ["TestFakeIBMBackend"]

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance # type: ignore

from qickit.backend import Backend, FakeBackend
from qickit.circuit import Circuit, CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit
from tests.backend import Template


def cosine_similarity(h1: dict[str, int],
                      h2: dict[str, int]) -> float:
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


class MockIBMBackend(FakeBackend):
    """ `MockIBMBackend` is the mock class for the `qickit.backend.FakeIBMBackend` class.
    """
    def __init__(self) -> None:
        self._qc_framework = QiskitCircuit

    @Backend.backendmethod
    def get_statevector(self,
                        circuit: Circuit) -> NDArray[np.complex128]:
        # Get the counts of the circuit
        counts = self.get_counts(circuit, num_shots=2**(2*circuit.num_qubits))

        # Create the state vector from the counts
        state_vector = np.zeros(2**circuit.num_qubits, dtype=np.complex128)

        # Set the state vector elements for the states in the counts
        for state, count in counts.items():
            state_vector[int(state, 2)] = np.sqrt(count)

        # Normalize the state vector
        state_vector /= np.linalg.norm(state_vector)

        return state_vector

    @Backend.backendmethod
    def get_operator(self,
                     circuit: Circuit) -> NDArray[np.complex128]:
        # This is a mock implementation, so we will return the expected result without
        # using the api call to the IBM Quantum service
        return np.array([[0.70710678+0.j, 0.70710678+0.j, 0.+0.j, 0.+0.j],
                         [0.+0.j, 0.+0.j, 0.70710678+0.j, -0.70710678+0.j],
                         [0.+0.j, 0.+0.j, 0.70710678+0.j, 0.70710678+0.j],
                         [0.70710678+0.j, -0.70710678+0.j, 0.+0.j, 0.+0.j]])

    @Backend.backendmethod
    def get_counts(self,
                   circuit: Circuit,
                   num_shots: int = 1024) -> dict[str, int]:
        # Create a copy of the circuit as measurement is applied inplace
        circuit = circuit.copy()

        # Extract the quasi-probability distribution from the first result
        # This is a mock implementation, so we will return the expected result without
        # using the api call to the IBM Quantum service
        quasi_dist = {1: 0.0263671875, 3: 0.484375, 2: 0.0185546875, 0: 0.470703125}

        # Convert the quasi-probability distribution to counts
        counts = {bin(k)[2:].zfill(circuit.num_qubits): int(v * num_shots) \
                  for k, v in quasi_dist.items()}

        # Fill the counts dict with zeros for the missing states
        counts = {f'{i:0{circuit.num_qubits}b}': counts.get(f'{i:0{circuit.num_qubits}b}', 0) \
                  for i in range(2**circuit.num_qubits)}

        # Sort the counts by their keys (basis states)
        counts = dict(sorted(counts.items()))

        return counts


class TestFakeIBMBackend(Template):
    """ `TestFakeIBMBackend` is the tester for the `FakeIBMBackend` class.
    """
    def test_init(self) -> None:
        """ Test the initialization of the backend.
        """
        # Define the `qickit.backend.MockIBMBackend` instance
        backend = MockIBMBackend()

    def test_get_counts(self) -> None:
        """ Test the `.get_counts()` method.
        """
        # Define the `qickit.backend.MockIBMBackend` instance
        backend = MockIBMBackend()

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
        """ Test the `.get_statevector()` method.
        """
        # Define the `qickit.backend.MockIBMBackend` instance
        backend = MockIBMBackend()

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

    def test_get_unitary(self) -> None:
        """ Test the `.get_unitary()` method.
        """
        # Define the `qickit.backend.MockIBMBackend` instance
        backend = MockIBMBackend()

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
        output_operator = np.array([[0.70710678+0.j, 0.70710678+0.j, 0.+0.j, 0.+0.j],
                                    [0.+0.j, 0.+0.j, 0.70710678+0.j, -0.70710678+0.j],
                                    [0.+0.j, 0.+0.j, 0.70710678+0.j, 0.70710678+0.j],
                                    [0.70710678+0.j, -0.70710678+0.j, 0.+0.j, 0.+0.j]])

        # Ensure the resulting statevectors are close enough (99 percent fidelity)
        assert 1 - distance.cosine(cirq_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(pennylane_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(qiskit_operator.flatten(), output_operator.flatten()) > 0.99
        assert 1 - distance.cosine(tket_operator.flatten(), output_operator.flatten()) > 0.99