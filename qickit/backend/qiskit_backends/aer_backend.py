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

__all__ = ["AerBackend"]

import copy
import numpy as np
from numpy.typing import NDArray

# Qiskit imports
from qiskit.primitives import BackendSampler # type: ignore
from qiskit_aer.aerprovider import AerSimulator # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore

# Import `qickit.circuit.Circuit` instances
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.Backend` class
from qickit.backend import Backend


class AerBackend(Backend):
    """ `qickit.backend.AerBackend` is the class for running `qickit.circuit.Circuit`
    instances on Aer.

    Attributes
    ----------
    `_qc_framework` : Type[QiskitCircuit]
        The quantum computing framework to use.
    """
    def __init__(self) -> None:
        self._qc_framework = QiskitCircuit

    @Backend.backendmethod
    def get_statevector(self,
                        circuit: Circuit) -> NDArray[np.complex128]:
        # Run the circuit to get the statevector
        state_vector = Statevector(circuit.circuit).data

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
                   num_shots: int) -> dict[str, int]:
        # Create a copy of the circuit as measurement is applied inplace
        circuit = copy.deepcopy(circuit)

        # Assert the number of shots is valid (an integer greater than 0)
        if not isinstance(num_shots, int) or num_shots <= 0:
            raise ValueError("The number of shots must be a positive integer.")

        # Measure the qubits
        if not circuit.measured:
            circuit.measure(list(range(circuit.num_qubits)))

        # Define the backend to run the circuit on
        backend = BackendSampler(AerSimulator())

        # Run the circuit on the backend to generate the result
        result = backend.run(circuit.circuit, shots=num_shots, seed_simulator=0).result()

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