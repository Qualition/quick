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

""" Abstract base class for converting quantum circuits from other
quantum computing frameworks to quick.
"""

from __future__ import annotations

__all__ = ["FromFramework"]

from abc import ABC, abstractmethod
from typing import Any, Type, TYPE_CHECKING

import quick
if TYPE_CHECKING:
    from quick.circuit import Circuit


class FromFramework(ABC):
    """ `quick.circuit.from_framework.FromFramework` is an abstract class for converting quantum circuits from other
    quantum computing frameworks to `quick.circuit.Circuit` class.

    Parameters
    ----------
    `output_framework` : type[Circuit]
        The quantum computing framework to convert the quantum circuit to.

    Attributes
    ----------
    `output_framework` : type[Circuit]
        The quantum computing framework to convert the quantum circuit to.

    Raises
    ------
    `TypeError`
        - If the `output_framework` is not a subclass of `quick.circuit.Circuit`.

    Usage
    -----
    >>> from_framework = FromFramework(output_framework=QiskitCircuit)
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:
        """ Initialize a `quick.circuit.from_framework.FromFramework` instance.
        """
        if not issubclass(output_framework, quick.circuit.Circuit):
            raise TypeError(
                "The `output_framework` must be a subclass of `quick.circular.Circuit`. "
                f"Received: {type(output_framework)}"
            )

        self.output_framework = output_framework

    @abstractmethod
    def convert(
            self,
            circuit: Any
        ) -> Circuit:
        """ Convert a quantum circuit from another quantum computing framework to
        `quick.circuit.Circuit` class.

        Parameters
        ----------
        `circuit` : Any
            The quantum circuit to be converted.

        Returns
        -------
        `quick_circuit` : quick.circuit.Circuit
            The converted quantum circuit.

        Raises
        ------
        `NotImplementedError`
            - If a gate in the quantum circuit is not supported by quick.

        Usage
        -----
        >>> quick_circuit = from_framework.convert(circuit)
        """