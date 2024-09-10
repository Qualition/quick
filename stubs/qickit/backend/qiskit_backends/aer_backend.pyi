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
from numpy.typing import NDArray
from qickit.backend import NoisyBackend
from qickit.circuit import Circuit

__all__ = ["AerBackend"]

class AerBackend(NoisyBackend):
    def __init__(self, single_qubit_error: float=0.0, two_qubit_error: float=0.0, device: str="CPU") -> None: ...
    def get_statevector(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    def get_operator(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    def get_counts(self, circuit: Circuit, num_shots: int=1024) -> dict[str, int]: ...
