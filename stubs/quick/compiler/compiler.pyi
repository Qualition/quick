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

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from quick.circuit import Circuit
from quick.optimizer import Optimizer
from quick.primitives import Bra, Ket, Operator
from quick.synthesis.statepreparation import StatePreparation
from quick.synthesis.unitarypreparation import UnitaryPreparation
from typing import Type, TypeAlias

__all__ = ["Compiler"]

PRIMITIVE: TypeAlias = Bra | Ket | Operator | NDArray[np.complex128]
PRIMITIVES: TypeAlias = list[tuple[PRIMITIVE, Sequence[int]]]


class Compiler:
    circuit_framework: Type[Circuit]
    state_prep: Type[StatePreparation]
    unitary_prep: Type[UnitaryPreparation]
    optimizer: Optimizer
    def __init__(self, circuit_framework: Circuit, state_prep: type[StatePreparation] = ..., unitary_prep: type[UnitaryPreparation] = ..., mlir: bool = True) -> None: ...
    def state_preparation(self, state: NDArray[np.complex128] | Bra | Ket) -> Circuit: ...
    def unitary_preparation(self, unitary: NDArray[np.complex128] | Operator) -> Circuit: ...
    def optimize(self, circuit: Circuit) -> Circuit: ...
    @staticmethod
    def _check_primitive(primitive: PRIMITIVE) -> None: ...
    @staticmethod
    def _check_primitive_qubits(primitive: PRIMITIVE, qubit_indices: Sequence[int]) -> None: ...
    @staticmethod
    def _check_primitives(primitives: PRIMITIVES) -> None: ...
    def _compile_primitive(self, primitive: PRIMITIVE) -> Circuit: ...
    def compile(self, primitives: PRIMITIVE | PRIMITIVES) -> Circuit: ...
