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
from quick.backend import FakeBackend
from quick.circuit import Circuit
from qiskit_ibm_runtime import QiskitRuntimeService

__all__ = ["FakeIBMBackend"]

class FakeIBMBackend(FakeBackend):
    def __init__(self, hardware_name: str, qiskit_runtime: QiskitRuntimeService, device: str="CPU") -> None: ...
    def get_statevector(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    def get_operator(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    def get_counts(self, circuit: Circuit, num_shots: int=1024) -> dict[str, int]: ...
