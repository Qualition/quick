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

from __future__ import annotations

__all__ = ["Collection"]

from typing import (Iterator, overload, Protocol, TypeVar,
                    TypeAlias, Self, runtime_checkable)


T = TypeVar("T")

@runtime_checkable
class Collection(Protocol[T]):
  def __len__(self) -> int:
    ...
  def __iter__(self) -> Iterator[T]:
    ...
  @overload
  def __getitem__(self, idx: int) -> T:
    ...
  @overload
  def __getitem__(self, idx: slice) -> Self:
    ...
  @overload
  def __setitem__(self, idx: int, value: T) -> None:
    ...
  @overload
  def __setitem__(self, idx: slice, value: Self) -> None:
    ...
  def __add__(self, other: Self) -> Self:
    ...
  def __mul__(self, other: int) -> Self:
    ...


# `NestedCollection` is a type alias that represents a collection of elements
# of type T or a collection of collections of elements of type T.
NestedCollection: TypeAlias = Collection[T] | Collection["NestedCollection[T]"]