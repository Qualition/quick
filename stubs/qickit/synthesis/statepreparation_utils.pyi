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

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray as NDArray


def gray_code(index: int) -> int: ...
def compute_alpha_y(magnitude: NDArray[np.float64], k: int, j: int) -> float: ...
def compute_alpha_z(phase: NDArray[np.float64], k: int, j: int) -> float: ...
def compute_m(k: int) -> NDArray[np.float64]: ...
def compute_control_indices(index: int) -> list[int]: ...
def bloch_angles(pair_of_complex: Sequence[complex]) -> tuple: ...
def rotations_to_disentangle(local_param: NDArray[np.complex128]) -> tuple: ...
