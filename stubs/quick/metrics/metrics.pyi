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
from numpy.typing import NDArray

__all__ = [
    "get_entanglements",
    "calculate_shannon_entropy",
    "calculate_entanglement_entropy"
]

def calculate_entanglement_range(statevector: NDArray[np.complex128]) -> list[tuple[int, int]]: ...
def calculate_shannon_entropy(statevector: NDArray[np.complex128]) -> float: ...
def calculate_entanglement_entropy(statevector: NDArray[np.complex128]) -> float: ...
