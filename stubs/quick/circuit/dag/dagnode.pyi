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

from dataclasses import dataclass
from typing import Hashable

__all__ = ["DAGNode"]

@dataclass
class DAGNode:
    name: Hashable = ...
    parents: set[DAGNode] = ...
    children: set[DAGNode] = ...
    def to(self, child: DAGNode) -> None: ...
    @property
    def depth(self) -> int: ...
    def generate_paths(self) -> set[tuple[Hashable]]: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __init__(self, name=..., parents=..., children=...) -> None: ...
