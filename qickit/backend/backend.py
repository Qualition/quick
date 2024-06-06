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

__all__ = ["Backend", "NoisyBackend", "FakeBackend"]

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from typing import Type

# Import `qickit.circuit.Circuit` instances
from qickit.circuit import Circuit


class Backend(ABC):
    """ `qickit.backend.Backend` is the abstract base class for
    running `qickit.circuit.Circuit` instances. This provides both
    GPU and NISQ hardware support.

    Attributes
    ----------
    `_qc_framework` : Type[qickit.circuit.Circuit]
        The quantum computing framework to use.

    Usage
    -----
    >>> backend = Backend()
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.backend.Backend` instance.
        """
        self._qc_framework: Type[Circuit]

    @staticmethod
    def backendmethod(method):
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
        def wrapped(instance, circuit: Circuit, **kwargs):
            # Ensure the circuit is of type `qickit.circuit.Circuit`
            if not isinstance(circuit, Circuit):
                raise TypeError(f"The circuit must be of type `qickit.circuit.Circuit`, not {type(circuit)}.")

            # Check if the instance has attribute `_max_num_queues`, and if so, ensure the circuit is compatible
            # NOTE: This is used by `FakeBackend` instances as they emulate real-world hardware
            if hasattr(instance, "_max_num_qubits") and circuit.num_qubits > instance._max_num_qubits:
                raise ValueError(f"The maximum number of qubits supported by the backend is {instance._max_num_qubits}.")

            # If the circuit passed is an instance of `qickit.circuit.Circuit`,
            # then ensure it is compatible with the backend framework
            if not isinstance(circuit, instance._qc_framework):
                circuit = circuit.convert(instance._qc_framework)

            return method(instance, circuit, **kwargs)

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


class NoisyBackend(Backend):
    """ `qickit.backend.NoisyBackend` is the abstract base class
    for running `qickit.circuit.Circuit` instances on noisy quantum
    devices.

    Parameters
    ----------
    `single_qubit_error` : float
        The error rate for single-qubit gates.
    `two_qubit_error` : float
        The error rate for two-qubit gates.

    Attributes
    ----------
    `_qc_framework` : Type[qickit.circuit.Circuit]
        The quantum computing framework to use.
    `single_qubit_error` : float
        The error rate for single-qubit gates.
    `two_qubit_error` : float
        The error rate for two-qubit gates.

    Usage
    -----
    >>> backend = NoisyBackend(single_qubit_error=0.01, two_qubit_error=0.02)
    """
    def __init__(self,
                 single_qubit_error: float,
                 two_qubit_error: float) -> None:
        """ Initialize a `qickit.backend.NoisyBackend` instance.
        """
        self._qc_framework: Type[Circuit]
        self.single_qubit_error = single_qubit_error
        self.two_qubit_error = two_qubit_error


class FakeBackend(Backend):
    """ `qickit.backend.FakeBackend` is the abstract base class
    for running `qickit.circuit.Circuit` instances on real quantum
    hardware emulators.

    Attributes
    ----------
    `_qc_framework` : Type[qickit.circuit.Circuit]
        The quantum computing framework to use.
    `_backend_name` : str
        The name of the backend to use.
    `_max_num_qubits` : int
        The maximum number of qubits supported by the backend.

    Usage
    -----
    >>> backend = FakeBackend()
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.backend.FakeBackend` instance.
        """
        self._qc_framework: Type[Circuit]
        self._backend_name: str
        self._max_num_qubits: int