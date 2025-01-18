# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/quick/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Wrapper class for the Aer quantum simulator backend.
"""

from __future__ import annotations

__all__ = ["AerBackend"]

import numpy as np
from numpy.typing import NDArray
import warnings

from qiskit.primitives import BackendSamplerV2 as BackendSampler # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore
from qiskit_aer import AerSimulator # type: ignore
import qiskit_aer.noise as noise # type: ignore

from quick.circuit import Circuit, QiskitCircuit
from quick.backend import Backend, NoisyBackend


class AerBackend(NoisyBackend):
    """ `quick.backend.AerBackend` is the class for running `quick.circuit.Circuit`
    instances on Aer. This supports ideal and noisy simulations, and allows for running
    on both CPU and GPU.

    Notes
    -----
    Aer is a high-performance simulator for quantum circuits developed by IBM Quantum.
    It supports both ideal and noisy simulations, and can run on both CPU and GPU.

    For more information, see https://qiskit.github.io/qiskit-aer/.

    This
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
    `_qc_framework` : type[quick.circuit.QiskitCircuit]
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
        - If the device is not "CPU" or "GPU".
        - If the single-qubit error rate is not between 0 and 1.
        - If the two-qubit error rate is not between 0 and 1.

    Usage
    -----
    >>> from quick.backend import AerBackend
    >>> aer_backend = AerBackend(single_qubit_error=0.01, two_qubit_error=0.02, device="GPU")
    """
    def __init__(
            self,
            single_qubit_error: float=0.0,
            two_qubit_error: float=0.0,
            device: str="CPU"
        ) -> None:

        super().__init__(
            single_qubit_error=single_qubit_error,
            two_qubit_error=two_qubit_error,
            device=device
        )

        self._qc_framework = QiskitCircuit

        if self.noisy:
            # Define depolarizing quantum errors (only on U3 and CX gates)
            single_qubit_error = noise.depolarizing_error(self.single_qubit_error, num_qubits=1) # type: ignore
            two_qubit_error = noise.depolarizing_error(self.two_qubit_error, num_qubits=2) # type: ignore

            # Add errors to the noise model
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(single_qubit_error, ["u", "u3"])
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])

        # Define the backend to run the circuit on
        # (based on device chosen and if noisy simulation is required)
        available_devices: list[str] = AerSimulator().available_devices() # type: ignore
        if "GPU" in available_devices and device == "GPU":
            if self.noisy:
                self._counts_backend = BackendSampler(backend=AerSimulator(device="GPU", noise_model=noise_model))
                self._op_backend = AerSimulator(device="GPU", method="unitary", noise_model=noise_model)
            else:
                self._counts_backend = BackendSampler(backend=AerSimulator(device="GPU"))
                self._op_backend = AerSimulator(device="GPU", method="unitary")
        else:
            if self.device == "GPU" and "GPU" not in available_devices:
                warnings.warn("Warning: GPU acceleration is not available. Defaulted to CPU.")
            if self.noisy:
                self._counts_backend = BackendSampler(backend=AerSimulator(noise_model=noise_model))
                self._op_backend = AerSimulator(method="unitary", noise_model=noise_model)
            else:
                self._counts_backend = BackendSampler(backend=AerSimulator())
                self._op_backend = AerSimulator(method="unitary")

    @Backend.backendmethod
    def get_statevector(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:

        # Create a copy of the circuit as `.remove_measurements()` is applied inplace
        circuit = circuit.copy()

        # Transpile the circuit to the backend
        # This is to counter https://github.com/Qiskit/qiskit/issues/13162
        circuit.transpile()

        # For circuits with more than 10 qubits or so, it's more efficient to use
        # AerSimulator to generate the statevector
        if circuit.num_qubits < 10 and self.noisy is False:
            circuit.remove_measurements(inplace=True)
            return Statevector(circuit.circuit).data

        else:
            # Measure all qubits to get the statevector
            circuit.measure_all()
            counts = self.get_counts(circuit, num_shots=2**(2*circuit.num_qubits))
            state_vector: NDArray[np.complex128] = np.zeros(2**circuit.num_qubits, dtype=np.complex128)
            for state, count in counts.items():
                state_vector[int(state, 2)] = np.sqrt(count)
            state_vector /= np.linalg.norm(state_vector)

        return state_vector

    @Backend.backendmethod
    def get_operator(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:

        # Create a copy of the circuit as `.remove_measurements()` is applied inplace
        circuit = circuit.copy()

        circuit.remove_measurements(inplace=True)

        # Transpile the circuit to the backend
        # This is to counter https://github.com/Qiskit/qiskit/issues/13162
        circuit.transpile()

        # For circuits with more than 10 qubits or so, it's more efficient to use
        # AerSimulator to generate the operator
        if circuit.num_qubits < 10 and self.noisy is False:
            operator = Operator(circuit.circuit).data

        else:
            circuit.circuit.save_unitary() # type: ignore
            operator = self._op_backend.run(circuit.circuit).result().get_unitary()

        return np.array(operator, dtype=np.complex128)

    @Backend.backendmethod
    def get_counts(
            self,
            circuit: Circuit,
            num_shots: int=1024
        ) -> dict[str, int]:

        if len(circuit.measured_qubits) == 0:
            raise ValueError("The circuit must have at least one measured qubit.")

        # Create a copy of the circuit as `.transpile()` is applied inplace
        circuit = circuit.copy()

        # Transpile the circuit to the backend
        # This is to counter https://github.com/Qiskit/qiskit/issues/13162
        circuit.transpile()

        # Run the circuit on the backend to generate the result
        # Transpile the circuit to the backend
        # This is to counter https://github.com/Qiskit/qiskit/issues/13162
        result = self._counts_backend.run([circuit.circuit], shots=num_shots).result()

        # Extract the counts from the result
        counts = result[0].join_data().get_counts() # type: ignore

        partial_counts = {}

        # Parse the binary strings to filter out the unmeasured qubits
        for key in counts.keys():
            new_key = ''.join(key[::-1][i] for i in range(len(key)) if i in circuit.measured_qubits)
            partial_counts[new_key[::-1]] = counts[key]

        counts = partial_counts

        # Fill the counts dict with zeros for the missing states
        num_qubits_to_measure = len(circuit.measured_qubits)

        counts = {
            f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) \
            for i in range(2**num_qubits_to_measure)
        }

        # Sort the counts by their keys (basis states)
        counts = dict(sorted(counts.items()))

        return counts