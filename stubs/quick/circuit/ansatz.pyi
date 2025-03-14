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
from quick.circuit import Circuit

__all__ = ["Ansatz"]

Params = list[list[float] | float] | list[float]

class Ansatz:
    ansatz: Circuit
    ignore_global_phase: bool = True
    def __init__(self, ansatz: Circuit, ignore_global_phase: bool = True) -> None: ...
    @property
    def thetas(self) -> Params: ...
    @thetas.setter
    def thetas(self, theta_values: Params) -> None: ...
    @property
    def num_params(self) -> int: ...
    @property
    def num_parameterized_gates(self) -> int: ...
    @property
    def is_parameterized(self) -> bool: ...
