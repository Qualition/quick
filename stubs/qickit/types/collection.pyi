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

from typing import Iterator, Protocol, Self, TypeVar, overload

__all__ = ["Collection"]

T = TypeVar("T", covariant=True)

class Collection(Protocol[T]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> Self: ...
    def __add__(self, other: Self) -> Self: ...
    def __mul__(self, other: int) -> Self: ...
