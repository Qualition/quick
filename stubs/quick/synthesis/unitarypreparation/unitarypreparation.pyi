# Copyright 2023-2025 Qualition Computing LLC.
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
import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from quick.circuit import Circuit
from quick.primitives import Operator
from typing import Type

__all__ = ["UnitaryPreparation"]

class UnitaryPreparation(ABC, metaclass=abc.ABCMeta):
    output_framework: Type[Circuit]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    def prepare_unitary(self, unitary: NDArray[np.complex128] | Operator) -> Circuit: ...
    @abstractmethod
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...
