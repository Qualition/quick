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
from quick.circuit.from_framework import FromFramework
from qiskit import QuantumCircuit
from typing import Callable

__all__ = ["FromQiskit"]

class FromQiskit(FromFramework):
    gate_mapping: dict[str, Callable]
    skip_gates: list[str]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    def extract_params(self, circuit: QuantumCircuit) -> list[dict]: ...
    def convert(self, circuit: QuantumCircuit) -> Circuit: ...
