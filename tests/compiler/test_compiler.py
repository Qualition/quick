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

__all__ = ["Template"]

from abc import ABC, abstractmethod


class Template(ABC):
    """ `tests.compiler.Template` is the template for creating compiler testers.
    """
    @abstractmethod
    def test_init(self) -> None:
        """ Test the initialization of the compiler.
        """

    @abstractmethod
    def test_state_preparation(self) -> None:
        """ Test the `compiler.state_preparation()` method.
        """

    @abstractmethod
    def test_unitary_preparation(self) -> None:
        """ Test the `compiler.unitary_preparation()` method.
        """

    @abstractmethod
    def test_compile_bra(self) -> None:
        """ Test the compilation of a `qickit.primitives.Bra` instance.
        """

    @abstractmethod
    def test_compile_ket(self) -> None:
        """ Test the compilation of a `qickit.primitives.Ket` instance.
        """

    @abstractmethod
    def test_compile_operator(self) -> None:
        """ Test the compilation of a `qickit.primitives.Operator` instance.
        """

    @abstractmethod
    def test_compile_ndarray(self) -> None:
        """ Test the compilation of a `numpy.ndarray` instance.
        """

    @abstractmethod
    def test_compile_invalid_primitive(self) -> None:
        """ Test the fail compilation of an invalid primitive.
        """