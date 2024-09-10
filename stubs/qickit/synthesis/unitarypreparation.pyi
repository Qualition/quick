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

import abc
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numpy.typing import NDArray
from qickit.circuit import Circuit
from qickit.primitives import Operator
from qiskit_ibm_runtime import QiskitRuntimeService
from typing import Type


__all__ = ["UnitaryPreparation", "QiskitUnitaryTranspiler"]

class UnitaryPreparation(ABC, metaclass=abc.ABCMeta):
    output_framework: Type[Circuit]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    @abstractmethod
    def prepare_unitary(self, unitary: NDArray[np.complex128] | Operator) -> Circuit: ...
    @abstractmethod
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...


class QiskitUnitaryTranspiler(UnitaryPreparation):
    ai_transpilation: bool
    service: QiskitRuntimeService | None
    backend_name: str | None
    def __init__(self, output_framework: type[Circuit], ai_transpilation: bool = False, service: QiskitRuntimeService | None = None, backend_name: str | None = None) -> None: ...
    def prepare_unitary(self, unitary: NDArray[np.complex128] | Operator) -> Circuit: ...
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...
