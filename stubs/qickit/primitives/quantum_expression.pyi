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

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
from qickit.backend import Backend
from qickit.primitives import Bra, Ket, Operator
from qickit.synthesis.statepreparation import StatePreparation
from qickit.synthesis.unitarypreparation import UnitaryPreparation

__all__ = ["QuantumExpression"]

class QuantumExpression:
    backend: Incomplete
    state_preparation_method: Incomplete
    unitary_preparation_method: Incomplete
    def __init__(self, expression: list[Bra | Ket | Operator], backend: Backend, state_preparation_method: type[StatePreparation] | None = None, unitary_preparation_method: type[UnitaryPreparation] | None = None) -> None: ...
    expression: Incomplete
    def check_expression(self, expression: list[Bra | Ket | Operator]) -> None: ...
    def evaluate(self) -> NDArray[np.complex128]: ...
