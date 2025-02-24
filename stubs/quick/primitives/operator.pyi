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
import quick.primitives.ket as ket
from numpy.typing import NDArray
from typing import overload, SupportsFloat, TypeAlias

__all__ = ["Operator"]

Scalar: TypeAlias = SupportsFloat | complex

class Operator:
    label: str
    data: NDArray[np.complex128]
    shape: tuple[int, int]
    num_qubits: int
    def __init__(self, data: NDArray[np.complex128], label: str | None = None) -> None: ...
    @staticmethod
    def is_unitary(data: NDArray[np.number]) -> None: ...
    @overload
    def __mul__(self, other: Scalar) -> Operator: ...
    @overload
    def __mul__(self, other: ket.Ket) -> ket.Ket: ...
    @overload
    def __mul__(self, other: Operator) -> Operator: ...
    def __rmul__(self, other: Scalar) -> Operator: ...
