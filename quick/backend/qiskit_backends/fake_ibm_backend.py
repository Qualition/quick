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

""" Wrapper class for the IBM quantum hardware emulator backend.
"""

from __future__ import annotations

__all__ = ["FakeIBMBackend"]

import numpy as np
from numpy.typing import NDArray

from qiskit.primitives import BackendSamplerV2 as BackendSampler # type: ignore
from qiskit_aer import AerSimulator # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
from qiskit.quantum_info import Operator # type: ignore

from quick.circuit import Circuit, QiskitCircuit
from quick.backend import Backend, FakeBackend


class FakeIBMBackend(FakeBackend): # pragma: no cover
    """ `quick.backend.FakeIBMBackend` is the class for running
    `quick.circuit.Circuit` instances on an IBM hardware emulator.

    Notes
    -----
    This backend emulates an IBM quantum hardware by using the Aer simulator
    with the latest calibration results from the IBM backend.

    For more information, see https://docs.quantum.ibm.com/api/qiskit/providers_fake_provider.

    Parameters
    ----------
    `hardware_name` : str
        The name of the IBM hardware to emulate.
    `qiskit_runtime` : QiskitRuntimeService
        The Qiskit runtime service to use.
    `device` : str, optional, default="CPU"
        The device to use for simulating the circuit.

    Attributes
    ----------
    `_qc_framework` : type[quick.circuit.QiskitCircuit]
        The quantum computing framework to use.
    `_backend_name` : str
        The name of the IBM backend to emulate.
    `_max_num_qubits` : int
        The maximum number of qubits the IBM backend can handle.
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
        - If the specified IBM backend is not available.

    Warns
    -----
    UserWarning
        - If the device is "GPU" but GPU acceleration is not available.

    Usage
    -----
    >>> from qiskit.runtime import QiskitRuntimeService
    >>> qiskit_runtime = QiskitRuntimeService()
    >>> fake_backend = FakeIBMBackend(hardware_name="ibmq_melbourne",
    ...                               qiskit_runtime=qiskit_runtime,
    ...                               device="CPU")
    """
    def __init__(
            self,
            hardware_name: str,
            qiskit_runtime: QiskitRuntimeService,
            device: str="CPU"
        ) -> None:

        super().__init__(device=device)
        self._qc_framework = QiskitCircuit

        # Get the names of all available IBM backends for the qiskit runtime
        all_backend_names = [backend.name for backend in qiskit_runtime.backends()]

        # Check if the specified backend is available
        if hardware_name not in all_backend_names:
            raise ValueError(f"IBM backend '{hardware_name}' is not available.")

        # Get the specified backend from the runtime service
        self._backend_name = hardware_name
        backend = qiskit_runtime.backend(self._backend_name)

        # Set the maximum number of qubits the backend can handle
        self._max_num_qubits = backend.num_qubits

        # Generate a simulator that mimics the real quantum system with
        # the latest calibration results
        available_devices = AerSimulator.available_devices() # type: ignore
        if self.device == "GPU" and available_devices["GPU"]:
            self._counts_backend = BackendSampler(backend=AerSimulator.from_backend(backend, device="GPU"))
            self._op_backend = AerSimulator.from_backend(backend, device="GPU", method="unitary")
        else:
            if self.device == "GPU" and available_devices["GPU"] is None:
                print("Warning: GPU acceleration is not available. Defaulted to CPU.")
            self._counts_backend = BackendSampler(backend=AerSimulator.from_backend(backend))
            self._op_backend = AerSimulator.from_backend(backend, method="unitary")

    @Backend.backendmethod
    def get_statevector(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:

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
    def get_operator(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:

        # Run the circuit to get the operator
        # NOTE: Currently, the operator cannot be obtained with noise considered,
        # so the operator without noise is returned
        # NOTE: For circuits with more than 10 qubits or so, it's more efficient to use
        # AerSimulator to generate the operator
        if circuit.num_qubits < 10:
            operator = Operator(circuit.circuit).data

        else:
            # Create a copy of the circuit as `.save_unitary()` is applied inplace
            circuit = circuit.copy()

            # Save the unitary of the circuit
            circuit.circuit.save_unitary() # type: ignore

            # Run the circuit on the backend to generate the operator
            operator = self._op_backend.run(circuit.circuit).result().get_unitary()

        return np.array(operator, dtype=np.complex128)

    @Backend.backendmethod
    def get_counts(
            self,
            circuit: Circuit,
            num_shots: int=1024
        ) -> dict[str, int]:

        # Run the circuit on the backend to generate the result
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