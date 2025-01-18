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

import numpy as np
from numpy.typing import NDArray
from typing import Literal

__all__ = ["Gate"]

class Gate:
    name: str
    matrix: NDArray[np.complex128]
    num_qubits: int
    ordering: str
    def __init__(self, name: str, matrix: NDArray[np.complex128]) -> None: ...
    def adjoint(self) -> NDArray[np.complex128]: ...
    def control(self, num_control_qubits: int) -> Gate: ...
    def change_mapping(self, ordering: Literal["MSB", "LSB"]) -> None: ...
