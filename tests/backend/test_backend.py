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
    """ `tests.backend.Template` is the template for creating backend testers.
    """
    @abstractmethod
    def test_init(self) -> None:
        """ Test the initialization of the backend.
        """

    @abstractmethod
    def test_get_partial_counts(self) -> None:
        """ Test the `.get_counts()` method with partial measurement.
        """

    @abstractmethod
    def test_get_counts(self) -> None:
        """ Test the `.get_counts()` method.
        """

    @abstractmethod
    def test_get_statevector(self) -> None:
        """ Test the `.get_statevector()` method.
        """

    @abstractmethod
    def test_get_large_statevector(self) -> None:
        """ Test the `.get_statevector()` method with a large statevector.
        """

    @abstractmethod
    def test_get_unitary(self) -> None:
        """ Test the `.get_unitary()` method.
        """

    @abstractmethod
    def test_get_large_unitary(self) -> None:
        """ Test the `.get_unitary()` method with a large unitary.
        """