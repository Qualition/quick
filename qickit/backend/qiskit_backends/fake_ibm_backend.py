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

__all__ = ["FakeManila"]

import copy
import numpy as np
from numpy.typing import NDArray

# Qiskit imports
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
from qiskit_aer import AerSimulator # type: ignore
from qiskit.quantum_info import Operator # type: ignore

# Import `qickit.circuit.Circuit` instances
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.FakeBackend` class
from qickit.backend import Backend, FakeBackend

IBM_BACKENDS = {
    "ibmq_algiers": 27,
    "ibmq_almaden": 20,
    "ibmq_armonk": 1,
    "ibmq_athens": 5,
    "ibmq_auckland": 27,
    "ibmq_belem": 5,
    "ibmq_boeblingen": 20,
    "ibmq_bogota": 5,
    "ibmq_brisbane": 127,
    "ibmq_burlington": 5,
    "ibmq_cairo": 27,
    "ibmq_cambridge": 28,
    "ibmq_casablanca": 7,
    "ibmq_cusco": 127,
    "ibmq_essex": 5,
    "ibmq_geneva": 27,
    "ibmq_guadalupe": 16,
}


class FakeManila(FakeBackend):
    """ `qickit.backend.FakeManila` is the class for running
    `qickit.circuit.Circuit` instances on an IBM Manila emulator.
    """
    def __init__(self) -> None:
        self._qc_framework = QiskitCircuit
        self._backend_name = "ibmq_manila"
        self._max_num_qubits = 5

        # Get the specified backend from the runtime service
        service = QiskitRuntimeService()
        backend = service.get_backend(self._backend_name)

        # Generate a simulator that mimics the real quantum system with
        # the latest calibration results
        self._backend = AerSimulator.from_backend(backend)

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
        # Run the circuit to get the operator
        operator = Operator(circuit.circuit).data

        return operator

    @Backend.backendmethod
    def get_counts(self,
                   circuit: Circuit,
                   num_shots: int = 1024) -> dict[str, int]:
        # Create a copy of the circuit as measurement is applied inplace
        circuit = copy.deepcopy(circuit)

        # Assert the number of shots is valid (an integer greater than 0)
        if not isinstance(num_shots, int) or num_shots <= 0:
            raise ValueError("The number of shots must be a positive integer.")

        # Measure the qubits
        if not circuit.measured:
            circuit.measure(list(range(circuit.num_qubits)))

        # Run the circuit on the backend to generate the result
        result = self._backend.run(circuit.circuit, shots=num_shots, seed_simulator=0).result()

        # Extract the quasi-probability distribution from the first result
        quasi_dist = result.quasi_dists[0]

        # Convert the quasi-probability distribution to counts
        counts = {bin(k)[2:].zfill(circuit.num_qubits): int(v * num_shots) \
                  for k, v in quasi_dist.items()}

        # Fill the counts dict with zeros for the missing states
        counts = {f'{i:0{circuit.num_qubits}b}': counts.get(f'{i:0{circuit.num_qubits}b}', 0) \
                  for i in range(2**circuit.num_qubits)}

        # Sort the counts by their keys (basis states)
        counts = dict(sorted(counts.items()))

        return counts