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

""" Abstract Base Class for defining backends to run quantum circuits.
"""

from __future__ import annotations

__all__ = [
    "Backend",
    "NoisyBackend",
    "FakeBackend"
]

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from types import NotImplementedType
from typing import Type

from qickit.circuit import Circuit


class Backend(ABC):
    """ `qickit.backend.Backend` is the abstract base class for
    running `qickit.circuit.Circuit` instances. This class provides
    CPU, GPU and NISQ hardware support.

    Parameters
    ----------
    `device` : str, optional, default="CPU"
        The device to use for simulating the circuit.
        This can be either "CPU", or "GPU".

    Attributes
    ----------
    `device` : str
        The device to use for simulating the circuit.
        This can be either "CPU", or "GPU".
    `_qc_framework` : type[qickit.circuit.Circuit]
        The quantum computing framework to use.

    Raises
    ------
    ValueError
        - If the device is not "CPU" or "GPU".
    """
    def __init__(
            self,
            device: str="CPU"
        ) -> None:
        """ Initialize a `qickit.backend.Backend` instance.
        """
        if device not in ["CPU", "GPU"]:
            raise ValueError(f"Invalid device: {device}. Must be either 'CPU' or 'GPU'.")
        self.device = device
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
            - If the circuit is not of type `qickit.circuit.Circuit`.
        ValueError
            - If the number of shots is not a positive integer.
            - If the number of qubits in the circuit is greater than the maximum supported by the backend.

        Usage
        -----
        >>> @Backend.backendmethod
        ... def get_statevector(self, circuit: Circuit) -> NDArray[np.complex128]:
        ...     ...
        """
        @wraps(method)
        def wrapped(instance, circuit: Circuit, **kwargs):
            if not isinstance(circuit, Circuit):
                raise TypeError(f"The circuit must be of type `qickit.circuit.Circuit`, not {type(circuit)}.")

            if "num_shots" in kwargs:
                if not isinstance(kwargs.get("num_shots", 1), int) or kwargs["num_shots"] <= 0:
                    raise ValueError("The number of shots must be a positive integer.")

            # Check if the instance has attribute `_max_num_qubits`, and if so, ensure the circuit is compatible
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
    def get_statevector(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.

        Returns
        -------
        NDArray[np.complex128]
            The statevector of the circuit.

        Raises
        ------
        TypeError
            - The circuit is not of type `qickit.circuit.Circuit`.

        Usage
        -----
        >>> backed.get_statevector(circuit)
        """

    @abstractmethod
    def get_operator(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:
        """ Get the operator of the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.

        Returns
        -------
        NDArray[np.complex128]
            The operator of the circuit.

        Raises
        ------
        TypeError
            - The circuit is not of type `qickit.circuit.Circuit`.

        Usage
        -----
        >>> backed.get_operator(circuit)
        """

    @abstractmethod
    def get_counts(
            self,
            circuit: Circuit,
            num_shots: int=1024
        ) -> dict[str, int]:
        """ Get the counts of the backend.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.
        `num_shots` : int, optional, default=1024
            The number of shots to run.

        Returns
        -------
        dict[str, int]
            The counts of the circuit.

        Raises
        ------
        TypeError
            - The circuit is not of type `qickit.circuit.Circuit`.
        ValueError
            - The circuit must have at least one qubit that is measured.
            - The number of shots must be a positive integer.

        Usage
        -----
        >>> backed.get_counts(circuit, num_shots=1024)
        """

    def __str__(self) -> str:
        """ Return a string representation of the backend.

        Returns
        -------
        str
            The string representation of the backend.

        Usage
        -----
        >>> str(backend)
        """
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in self.__dict__.items())})"

    def __repr__(self) -> str:
        """ Return a string representation of the backend.

        Returns
        -------
        str
            The string representation of the backend.

        Usage
        -----
        >>> repr(backend)
        """
        return str(self)

    @classmethod
    def __subclasscheck__(cls, C) -> bool:
        """ Checks if a class is a `qickit.backend.Backend` if the class
        passed does not directly inherit from `qickit.backend.Backend`.

        Parameters
        ----------
        `C` : type
            The class to check if it is a subclass.

        Returns
        -------
        bool
            Whether or not the class is a subclass.
        """
        if cls is Backend:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return False

    @classmethod
    def __subclasshook__(cls, C) -> bool | NotImplementedType:
        """ Checks if a class is a `qickit.backend.Backend` if the class
        passed does not directly inherit from `qickit.backend.Backend`.

        Parameters
        ----------
        `C` : type
            The class to check if it is a subclass.

        Returns
        -------
        bool | NotImplementedType
            Whether or not the class is a subclass.
        """
        if cls is Backend:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return NotImplemented

    @classmethod
    def __instancecheck__(cls, C) -> bool:
        """ Checks if an object is a `qickit.backend.Backend` given its
        interface.

        Parameters
        ----------
        `C` : object
            The instance to check.

        Returns
        -------
        bool
            Whether or not the instance is a `qickit.backend.Backend`.
        """
        if cls is Backend:
            return all(hasattr(C, method) for method in list(cls.__dict__["__abstractmethods__"]))
        return False


class NoisyBackend(Backend, ABC):
    """ `qickit.backend.NoisyBackend` is the abstract base class
    for running `qickit.circuit.Circuit` instances on noisy quantum
    devices.

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
    `_qc_framework` : Type[qickit.circuit.Circuit]
        The quantum computing framework to use.
    `noisy` : bool
        Whether the simulation is noisy or not.

    Raises
    ------
    ValueError
        - If the device is not "CPU" or "GPU".
        - If the single-qubit error rate is not between 0 and 1.
        - If the two-qubit error rate is not between 0 and 1.
    """
    def __init__(
            self,
            single_qubit_error: float,
            two_qubit_error: float,
            device: str="CPU"
        ) -> None:
        """ Initialize a `qickit.backend.NoisyBackend` instance.
        """
        super().__init__(device=device)

        if single_qubit_error < 0 or single_qubit_error > 1:
            raise ValueError("The single-qubit error rate must be between 0 and 1.")
        self.single_qubit_error = single_qubit_error

        if two_qubit_error < 0 or two_qubit_error > 1:
            raise ValueError("The two-qubit error rate must be between 0 and 1.")
        self.two_qubit_error = two_qubit_error

        self.noisy = self.single_qubit_error > 0.0 or self.two_qubit_error > 0.0

        self._qc_framework: Type[Circuit]


class FakeBackend(Backend, ABC):
    """ `qickit.backend.FakeBackend` is the abstract base class
    for running `qickit.circuit.Circuit` instances on real quantum
    hardware emulators.

    Parameters
    ----------
    `device` : str, optional, default="CPU"
        The device to use for simulating the circuit.
        This can be either "CPU", or "GPU".

    Attributes
    ----------
    `device` : str
        The device to use for simulating the circuit.
        This can be either "CPU", or "GPU".
    `_qc_framework` : Type[qickit.circuit.Circuit]
        The quantum computing framework to use.
    `_backend_name` : str
        The name of the backend to use (usually the name of the backend being emulated).
    `_max_num_qubits` : int
        The maximum number of qubits supported by the backend.

    Raises
    ------
    ValueError
        - If the device is not "CPU" or "GPU".
    """
    def __init__(
            self,
            device: str="CPU"
        ) -> None:
        """ Initialize a `qickit.backend.FakeBackend` instance.
        """
        super().__init__(device=device)
        self._qc_framework: Type[Circuit]
        self._backend_name: str
        self._max_num_qubits: int

    @abstractmethod
    def get_statevector(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.

        Returns
        -------
        NDArray[np.complex128]
            The statevector of the circuit.

        Raises
        ------
        TypeError
            - The circuit is not of type `qickit.circuit.Circuit`.
        ValueError
            - The number of qubits in the circuit is greater than the maximum supported by the backend.

        Usage
        -----
        >>> backed.get_statevector(circuit)
        """

    @abstractmethod
    def get_operator(
            self,
            circuit: Circuit
        ) -> NDArray[np.complex128]:
        """ Get the operator of the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.

        Returns
        -------
        NDArray[np.complex128]
            The operator of the circuit.

        Raises
        ------
        TypeError
            - The circuit is not of type `qickit.circuit.Circuit`.
        ValueError
            - The number of qubits in the circuit is greater than the maximum supported by the backend.

        Usage
        -----
        >>> backed.get_operator(circuit)
        """

    @abstractmethod
    def get_counts(
            self,
            circuit: Circuit,
            num_shots: int=1024
        ) -> dict[str, int]:
        """ Get the counts of the backend.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to run.
        `num_shots` : int, optional, default=1024
            The number of shots to run.

        Returns
        -------
        dict[str, int]
            The counts of the circuit.

        Raises
        ------
        TypeError
            - The circuit is not of type `qickit.circuit.Circuit`.
        ValueError
            - The number of qubits in the circuit is greater than the maximum supported by the backend.
            - The circuit must have at least one qubit that is measured.
            - The number of shots must be a positive integer.

        Usage
        -----
        >>> backed.get_counts(circuit, num_shots=1024)
        """