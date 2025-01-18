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

import abc
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from quick.circuit import Circuit
from types import NotImplementedType

__all__ = ["Backend", "NoisyBackend", "FakeBackend"]

class Backend(ABC, metaclass=abc.ABCMeta):
    device: str
    def __init__(self, device: str="CPU") -> None: ...
    @staticmethod
    def backendmethod(method): ...
    @abstractmethod
    def get_statevector(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    @abstractmethod
    def get_operator(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    @abstractmethod
    def get_counts(self, circuit: Circuit, num_shots: int=1024) -> dict[str, int]: ...
    @classmethod
    def __subclasscheck__(cls, C) -> bool: ...
    @classmethod
    def __subclasshook__(cls, C) -> bool | NotImplementedType: ...
    @classmethod
    def __instancecheck__(cls, C) -> bool: ...

class NoisyBackend(Backend, metaclass=abc.ABCMeta):
    single_qubit_error: float
    two_qubit_error: float
    noisy: bool
    def __init__(self, single_qubit_error: float, two_qubit_error: float, device: str="CPU") -> None: ...

class FakeBackend(Backend, metaclass=abc.ABCMeta):
    def __init__(self, device: str="CPU") -> None: ...
