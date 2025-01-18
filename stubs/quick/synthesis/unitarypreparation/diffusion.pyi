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
from quick.circuit import Circuit
from quick.primitives.operator import Operator
from quick.synthesis.unitarypreparation import UnitaryPreparation
from typing import Any, Sequence

__all__ = ["Diffusion"]

class Diffusion(UnitaryPreparation):
    model: str
    prompt: str
    max_num_gates: int
    num_samples: int
    pipeline: Any
    def __init__(
            self,
            output_framework: type[Circuit],
            model: str="Floki00/qc_unitary_3qubit",
            prompt: str="Compile using: ['h', 'cx', 'z', 'ccx', 'swap']",
            max_num_gates: int=12,
            num_samples: int=128
        ) -> None: ...
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...
