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

__all__ = ["StatePreparation"]

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Bra, Ket


class StatePreparation(ABC):
    """ `qickit.synthesis.statepreparation.StatePreparation` is the class for preparing quantum states.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Usage
    -----
    >>> state_preparer = StatePreparation(output_framework=QiskitCircuit)
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:
        """ Initalize a State Preparation instance.
        """
        self.output_framework = output_framework

    @abstractmethod
    def prepare_state(
            self,
            state: NDArray[np.complex128] | Bra | Ket,
            compression_percentage: float = 0.0,
            index_type: Literal["row", "snake"]="row"
        ) -> Circuit:
        """ Prepare the quantum state.

        Parameters
        ----------
        `state` : NDArray[np.complex128] | qickit.primitives.Bra | qickit.primitives.Ket
            The quantum state to prepare.
        `compression_percentage` : float, optional, default=0.0
            Number between 0 an 100, where 0 is no compression and 100 all statevector values are 0.
        `index_type` : Literal["row", "snake"], optional, default="row"
            The indexing type for the data.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit that prepares the state.

        Raises
        ------
        TypeError
            If the state is not a numpy array or a Bra/Ket object.
        """