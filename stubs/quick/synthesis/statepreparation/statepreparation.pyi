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

import abc
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from quick.circuit import Circuit
from quick.primitives import Bra, Ket
from typing import Literal, Type

__all__ = ["StatePreparation"]

class StatePreparation(ABC, metaclass=abc.ABCMeta):
    output_framework: Type[Circuit]
    def __init__(self, output_framework: Type[Circuit]) -> None: ...
    def prepare_state(self, state: NDArray[np.complex128] | Bra | Ket, compression_percentage: float=0.0, index_type: str="row") -> Circuit: ...
    @abstractmethod
    def apply_state(
            self,
            circuit: Circuit,
            state: NDArray[np.complex128] | Bra | Ket,
            qubit_indices: int | Sequence[int],
            compression_percentage: float=0.0,
            index_type: Literal["row", "snake"]="row"
        ) -> Circuit: ...