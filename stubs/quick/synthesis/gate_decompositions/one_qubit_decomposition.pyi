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

import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence
from numpy.typing import NDArray
from quick.circuit import Circuit
from quick.primitives import Operator
from quick.synthesis.unitarypreparation import UnitaryPreparation
from typing import Literal

__all__ = ["OneQubitDecomposition"]

class OneQubitDecomposition(UnitaryPreparation):
    basis: Incomplete
    def __init__(self, circuit_framework: type[Circuit], basis: Literal["zyz", "u3"] = "u3") -> None: ...
    @staticmethod
    def params_zyz(U: NDArray[np.complex128]) -> tuple[float, tuple[float, float, float]]: ...
    @staticmethod
    def params_u3(U: NDArray[np.complex128]) -> tuple[float, tuple[float, float, float]]: ...
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...
