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

__all__ = ["StatePreparationTemplate"]

from abc import ABC, abstractmethod


class StatePreparationTemplate(ABC):
    """ `tests.synthesis.StatePreparationTemplate` is the template for creating
    state preparation testers.
    """
    @abstractmethod
    def test_init(self) -> None:
        """ Test the initialization of the state preparation.
        """

    @abstractmethod
    def test_prepare_state_ket(self) -> None:
        """ Test the preparation of the state from a `qickit.primitives.Ket` instance.
        """

    @abstractmethod
    def test_prepare_state_bra(self) -> None:
        """ Test the preparation of the state from a `qickit.primitives.Bra` instance.
        """

    @abstractmethod
    def test_prepare_state_ndarray(self) -> None:
        """ Test the preparation of the state from a numpy array.
        """