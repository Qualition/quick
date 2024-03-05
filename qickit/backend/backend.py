# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the GNU Version 3.0 (the "License");
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

__all__ = ['Backend', 'AerBackend']

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

# Qiskit imports
from qiskit_aer import AerSimulator, StatevectorSimulator

# Import `qickit.Circuit` instances
if TYPE_CHECKING:
    from qickit.circuit import *


class Backend(ABC):
    """ `qickit.Backend` is the class for running `qickit.Circuit` instances.
        This provides both GPU and NISQ hardware support.
    """
    def __init__(self) -> None:
        """ Initialize the backend.

        Parameters
        ----------
        `_qc_framework` (Circuit):
            The quantum computing framework to use.
        `_sv_only` (bool):
            If True, the backend will only support `get_statevector()`.
        """
        self._qc_framework: Circuit = None
        self._sv_only: bool = False

    @abstractmethod
    def get_statevector(self,
                        circuit: Circuit) -> NDArray[np.complex128]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `circuit` (Circuit):
            The circuit to run.

        Returns
        -------
        (NDArray[np.complex128]): The statevector of the circuit.
        """
        pass

    @abstractmethod
    def get_counts(self,
                   circuit: Circuit,
                   num_shots: int) -> dict:
        """ Get the counts of the backend.

        Parameters
        ----------
        `circuit` (Circuit):
            The circuit to run.
        `num_shots` (int):
            The number of shots to run.

        Returns
        -------
        (dict): The counts of the circuit.
        """
        pass


class AerBackend(Backend):
    """ `qickit.AerBackend` is the class for running `qickit.Circuit` instances on Aer.
    """
    def __init__(self) -> None:
        """ Initialize the backend.

        Parameters
        ----------
        `_qc_framework` (Circuit):
            The quantum computing framework to use.
        `_sv_only` (bool):
            If True, the backend will only support `get_statevector()`.
        """
        self._qc_framework = QiskitCircuit
        self._sv_only = False

    def get_statevector(self,
                        circuit: QiskitCircuit) -> NDArray[np.complex128]:
        """ Get the statevector of the circuit.

        Parameters
        ----------
        `circuit` (QiskitCircuit):
            The circuit to run.

        Returns
        -------
        `state_vector` (NDArray[np.complex128]): The statevector of the circuit.
        """
        # Assert the circuit type is supported
        try:
            isinstance(circuit, QiskitCircuit)
        except TypeError:
            raise TypeError("The circuit must be a QiskitCircuit.")

        # Define the backend
        backend = StatevectorSimulator()

        # Run the circuit
        state_vector = (backend.run(self.circuit.decompose(reps=100))).result().get_statevector()

        # Return the statevector
        return state_vector

    def get_counts(self,
                   circuit: QiskitCircuit,
                   num_shots: int) -> dict:
        """ Get the counts of the backend.

        Parameters
        ----------
        `circuit` (QiskitCircuit):
            The circuit to run.
        `num_shots` (int):
            The number of shots to run.

        Returns
        -------
        `counts` (dict): The counts of the circuit.
        """
        # Assert the circuit type is supported
        try:
            isinstance(circuit, QiskitCircuit)
        except TypeError:
            raise TypeError("The circuit must be a QiskitCircuit.")

        # Assert the number of shots is valid
        try:
            isinstance(num_shots, int) and num_shots > 0
        except ValueError:
            raise ValueError("The number of shots must be a positive integer.")

        # Define the backend
        backend = AerSimulator()

        # Run the circuit
        counts = backend.run(circuit, shots=num_shots).result().get_counts()

        # Return the counts
        return counts