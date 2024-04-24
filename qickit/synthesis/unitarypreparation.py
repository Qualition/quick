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

__all__ = ['UnitaryPreparation']

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from typing import Callable

# Import `qickit.circuit.Circuit` instances
from qickit.circuit import *


class UnitaryPreparation(ABC):
    """ `qickit.UnitaryPreparation` is the class for preparing quantum operators.

    Parameters
    ----------
    `circuit_framework` : qickit.circuit.Circuit
        The quantum circuit framework.
    """
    def __init__(self,
                 circuit_framework: Circuit) -> None:
        """ Initalize a Unitary Preparation instance.
        """
        # Define the QC framework
        self.circuit_framework = circuit_framework

        # Set the encoding schema
        self.encoder = self.prepare_unitary

    def check_unitary(self,
                      unitary: NDArray[np.complex128]) -> bool:
        """ Check if the input matrix is a valid unitary matrix.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128]
            The quantum unitary operator.

        Returns
        -------
        bool
            True if the input matrix is a valid unitary matrix, False otherwise.
        """
        # Check if the input matrix is a valid unitary matrix
        return np.allclose(np.eye(unitary.shape[0]), unitary.conj().T @ unitary)

    @staticmethod
    def unitarymethod(method: Callable) -> Callable:
        """ Decorator for unitary methods.

        Parameters
        ----------
        `method` : Callable
            The method to decorate.

        Returns
        -------
        `wrapper` : Callable
            The decorated method.
        """
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            # Check if the input matrix is a valid unitary matrix
            if not instance.check_unitary(args[1]):
                raise ValueError("Input matrix is not a unitary matrix.")

            # Run the method
            return method(instance, *args, **kwargs)

        # Return the decorated method
        return wrapper

    @abstractmethod
    def prepare_unitary(self,
                        unitary: NDArray[np.complex128]) -> Circuit:
        """ Prepare the quantum unitary operator.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128]
            The quantum unitary operator.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit for preparing the unitary operator.
        """
        pass