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

""" Abstract base class for defining optimizers to optimize quantum circuits.
"""

from __future__ import annotations

__all__ = ["Optimizer"]

from abc import ABC, abstractmethod

from qickit.circuit import Circuit


class Optimizer(ABC):
    """ `qickit.optimizer.Optimizer` is an abstract base class that defines the interface for
    optimizers that can be used to optimize quantum circuits. These optimizers include rewrite
    rules, circuit simplification, and other techniques to reduce the number of gates and qubits
    required to implement a quantum circuit.
    """
    @abstractmethod
    def optimize(
            self,
            circuit: Circuit
        ) -> Circuit:
        """ Optimize the given circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to be optimized.

        Returns
        -------
        `optimized_circuit` : qickit.circuit.Circuit
            The optimized circuit.

        Usage
        -----
        >>> optimized_circuit = optimizer.optimize(circuit)
        """