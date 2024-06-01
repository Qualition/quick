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

__all__ = ["Backend", "AerBackend"]

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Type

# Qiskit imports
from qiskit.primitives import BackendSampler # type: ignore
from qiskit_aer.aerprovider import AerSimulator # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore

# Import `qickit.circuit.Circuit` instances
from qickit.circuit import Circuit, QiskitCircuit


class Backend(ABC):
    """ `qickit.backend.Backend` is the class for running `qickit.circuit.Circuit`
    instances. This provides both GPU and NISQ hardware support.

    Attributes
    ----------
    `_qc_framework` : Type[qickit.circuit.Circuit]
        The quantum computing framework to use.
    `_sv_only` : bool
        If True, the backend will only support :func:`get_statevector()`.

    Usage
    -----
    >>> backend = Backend()
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.backend.Backend` instance.
        """
        self._qc_framework: Type[Circuit]
        self._sv_only: bool

    @staticmethod
    def backendmethod(method: Callable) -> Callable:
        """ Decorator for backend methods.

        Parameters
        ----------
        `method` : Callable
            The method to decorate.

        Returns
        -------
        `wrapper` : Callable
            The decorated method.

        Raises
        ------
        TypeError
            If the circuit is not of type `qickit.circuit.Circuit`.

        Usage
        -----
        >>> @Backend.backendmethod
        ... def get_statevector(self, circuit: Circuit) -> NDArray[np.complex128]:
        ...     ...
        """
        @wraps(method)
        def wrapped(instance, circuit: Circuit):
            # Ensure the circuit is of type `qickit.circuit.Circuit`
            if not isinstance(circuit, Circuit):
                raise TypeError(f"The circuit must be of type `qickit.Circuit`, not {type(circuit)}.")

            # If the circuit passed is an instance of `qickit.circuit.Circuit`,
            # then ensure it is compatible with the backend framework
            if not isinstance(circuit, instance._qc_framework):
                circuit = circuit.convert(instance._qc_framework)

            return method(instance, circuit)

        return wrapped

    @abstractmethod
    def get_statevector(self,
                        circuit: Circuit) -> NDArray[np.complex128]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.

        Returns
        -------
        NDArray[np.complex128]
            The statevector of the circuit.

        Usage
        -----
        >>> backed.get_statevector(circuit)
        """

    @abstractmethod
    def get_operator(self,
                     circuit: Circuit) -> NDArray[np.complex128]:
        """ Get the operator of the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.

        Returns
        -------
        NDArray[np.complex128]
            The operator of the circuit.

        Usage
        -----
        >>> backed.get_operator(circuit)
        """

    @abstractmethod
    def get_counts(self,
                   circuit: Circuit,
                   num_shots: int) -> dict[str, int]:
        """ Get the counts of the backend.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.
        `num_shots` : int
            The number of shots to run.

        Returns
        -------
        dict[str, int]
            The counts of the circuit.

        Raises
        ------
        ValueError
            If the number of shots is not a positive integer.

        Usage
        -----
        >>> backed.get_counts(circuit, num_shots=1024)
        """


class AerBackend(Backend):
    """ `qickit.AerBackend` is the class for running `qickit.Circuit` instances on Aer.
    """
    def __init__(self) -> None:
        self._qc_framework = QiskitCircuit
        self._sv_only = False

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
        # Assert the number of shots is valid (an integer greater than 0)
        if not isinstance(num_shots, int) or num_shots <= 0:
            raise ValueError("The number of shots must be a positive integer.")

        # Define the backend to run the circuit on
        backend: BackendSampler = BackendSampler(AerSimulator())

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