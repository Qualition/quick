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

from quick.circuit.gate_matrix import Gate

__all__ = [
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "S",
    "T",
    "RX",
    "RY",
    "RZ",
    "U3",
    "Phase"
]

class PauliX(Gate):
    def __init__(self) -> None: ...

class PauliY(Gate):
    def __init__(self) -> None: ...

class PauliZ(Gate):
    def __init__(self) -> None: ...

class Hadamard(Gate):
    def __init__(self) -> None: ...

class S(Gate):
    def __init__(self) -> None: ...

class T(Gate):
    def __init__(self) -> None: ...

class RX(Gate):
    def __init__(self, theta: float) -> None: ...

class RY(Gate):
    def __init__(self, theta: float) -> None: ...

class RZ(Gate):
    def __init__(self, theta: float) -> None: ...

class U3(Gate):
    def __init__(self, theta: float, phi: float, lam: float) -> None: ...

class Phase(Gate):
    def __init__(self, theta: float) -> None: ...
