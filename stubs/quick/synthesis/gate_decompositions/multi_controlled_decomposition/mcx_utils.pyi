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

from quick.circuit import Circuit

__all__ = ["CCX", "RCCX", "C3X", "C3SX", "RC3X", "C4X"]

def CCX(circuit: Circuit, control_indices: list[int], target_index: int) -> None: ...
def RCCX(circuit: Circuit, control_indices: list[int], target_index: int) -> None: ...
def C3X(circuit: Circuit, control_indices: list[int], target_index: int) -> None: ...
def C3SX(circuit: Circuit, control_indices: list[int], target_index: int) -> None: ...
def RC3X(circuit: Circuit, control_indices: list[int], target_index: int) -> None: ...
def C4X(circuit: Circuit, control_indices: list[int], target_index: int) -> None: ...
