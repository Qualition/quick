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
from qiskit.quantum_info import Statevector, Operator # type: ignore
from qiskit_aer import AerSimulator # type: ignore
import qiskit_aer.noise as noise # type: ignore

# Import `qickit.circuit.Circuit` instances
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.NoisyBackend` class
from qickit.backend import Backend, NoisyBackend


class AerBackend(NoisyBackend):
    """ `qickit.backend.AerBackend` is the class for running `qickit.circuit.Circuit`
    instances on Aer. This supports ideal and noisy simulations, and allows for running
    on both CPU and GPU.

    Parameters
    ----------
    `single_qubit_error` : float, optional, default=0.0
        The error rate for single-qubit gates.
    `two_qubit_error` : float, optional, default=0.0
        The error rate for two-qubit gates.
    `device` : str, optional, default="CPU"
        The device to use for simulating the circuit.
        This can be either "CPU", or "GPU".

    Attributes
    ----------
    `single_qubit_error` : float
        The error rate for single-qubit gates.
    `two_qubit_error` : float
        The error rate for two-qubit gates.
    `device` : str
        The device to use for simulating the circuit.
        This can be either "CPU", or "GPU".
    `_qc_framework` : type[qickit.circuit.QiskitCircuit]
        The quantum computing framework to use.
    `noisy` : bool
        Whether the simulation is noisy or not.
    `_counts_backend` : qiskit.primitives.BackendSampler
        The Aer simulator to use for generating counts.
    `_op_backend` : qiskit_aer.aerprovider.AerSimulator
        The Aer simulator to use for generating the operator.

    Raises
    ------
    ValueError
        If the device is not "CPU" or "GPU".
        If the single-qubit error rate is not between 0 and 1.
        If the two-qubit error rate is not between 0 and 1.
    """
    def __init__(self,
                 single_qubit_error: float=0.0,
                 two_qubit_error: float=0.0,
                 device: str="CPU") -> None:
        super().__init__(single_qubit_error=single_qubit_error,
                         two_qubit_error=two_qubit_error,
                         device=device)
        self._qc_framework = QiskitCircuit

        # If the noise rates are non-zero, then define the depolarizing quantum errors
        # and add them to the noise model
        if self.noisy:
            # Define depolarizing quantum errors (only on U3 and CX gates)
            single_qubit_error = noise.depolarizing_error(self.single_qubit_error, num_qubits=1)
            two_qubit_error = noise.depolarizing_error(self.two_qubit_error, num_qubits=2)

            # Add errors to the noise model
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(single_qubit_error, ["u", "u3"])
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])

        # Define the backend to run the circuit on
        # (based on device chosen and if noisy simulation is required)
        if "GPU" in AerSimulator().available_devices() and device == "GPU":
            if self.noisy:
                self._counts_backend = BackendSampler(AerSimulator(device="GPU", noise_model=noise_model))
                self._op_backend = AerSimulator(device="GPU", method="unitary", noise_model=noise_model)
            else:
                self._counts_backend = BackendSampler(AerSimulator(device="GPU"))
                self._op_backend = AerSimulator(device="GPU", method="unitary")
        else:
            if self.device == "GPU" and "GPU" not in AerSimulator().available_devices():
                print("Warning: GPU acceleration is not available. Defaulted to CPU.")
            if self.noisy:
                self._counts_backend = BackendSampler(AerSimulator(noise_model=noise_model))
                self._op_backend = AerSimulator(method="unitary", noise_model=noise_model)
            else:
                self._counts_backend = BackendSampler(AerSimulator())
                self._op_backend = AerSimulator(method="unitary")

    @Backend.backendmethod
    def get_statevector(self,
                        circuit: Circuit) -> NDArray[np.complex128]:
        # Run the circuit to get the statevector
        # NOTE: For circuits with more than 10 qubits or so, it's more efficient to use
        # AerSimulator to generate the statevector
        if circuit.num_qubits < 10 and self.noisy is False:
            # Remove the measurements from the circuit
            circuit.remove_measurements(inplace=True)
            state_vector = Statevector(circuit.circuit).data

        else:
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
        # Remove the measurements from the circuit
        circuit.remove_measurements(inplace=True)

        # Run the circuit to get the operator
        # NOTE: For circuits with more than 10 qubits or so, it's more efficient to use
        # AerSimulator to generate the operator
        if circuit.num_qubits < 10 and self.noisy is False:
            operator = Operator(circuit.circuit).data

        else:
            # Create a copy of the circuit as `.save_unitary()` is applied inplace
            circuit = copy.deepcopy(circuit)

            # Save the unitary of the circuit
            circuit.circuit.save_unitary() # type: ignore

            # Run the circuit on the backend to generate the operator
            operator = self._op_backend.run(circuit.circuit).result().get_unitary()

        return operator

    @Backend.backendmethod
    def get_counts(self,
                   circuit: Circuit,
                   num_shots: int) -> dict[str, int]:
        if not any(circuit.measured_qubits):
            raise ValueError("The circuit must have at least one measured qubit.")

        # Run the circuit on the backend to generate the result
        result = self._counts_backend.run(circuit.circuit, shots=num_shots, seed_simulator=0).result()

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