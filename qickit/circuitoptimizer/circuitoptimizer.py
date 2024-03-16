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

__all__ = ['CircuitOptimizer', 'CNOTOptimizer']

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

# Import `qickit.Circuit` instances
if TYPE_CHECKING:
    from qickit.circuit import *


class CircuitOptimizer(ABC):
    """ `qickit.CircuitOptimizer` is the class for optimizing `qickit.Circuit` instances.
    """
    def __init__(self) -> None:
        """ Initialize the circuit optimizer.
        """
        pass

    @abstractmethod
    def optimize(self,
                 circuit: Circuit) -> Circuit:
        """ Optimize the circuit.

        Parameters
        ----------
        `circuit` (Circuit):
            The circuit to optimize.

        Returns
        -------
        `Circuit`:
            The optimized circuit.
        """
        pass


class CNOTOptimizer(CircuitOptimizer):
    """ `qickit.CNOTOptimizer` is the class for optimizing `qickit.Circuit` instances by reducing the number of CNOT gates.
    """
    def optimize(self,
                 circuit: Circuit) -> Circuit:
        """ Optimize the circuit by reducing the number of CNOT gates.

        Parameters
        ----------
        `circuit` (Circuit):
            The circuit to optimize.

        Returns
        -------
        `Circuit`:
            The optimized circuit.
        """
        # TODO
        # Perform optimization

        # Return the circuit
        return circuit