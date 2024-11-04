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

__all__ = ["UnitaryPreparationTemplate"]

from abc import ABC, abstractmethod


class UnitaryPreparationTemplate(ABC):
    """ `tests.synthesis.test_unitarypreparation.UnitaryPreparationTemplate` is the template
    for creating unitary preparation testers.
    """
    @abstractmethod
    def test_init(self) -> None:
        """ Test the initialization of the unitary preparation.
        """

    @abstractmethod
    def test_init_invalid_output_framework(self) -> None:
        """ Test the initialization of the unitary preparation with an invalid output framework.
        """

    @abstractmethod
    def test_prepare_unitary_ndarray(self) -> None:
        """ Test the preparation of the unitary matrix from a numpy array.
        """

    @abstractmethod
    def test_prepare_unitary_operator(self) -> None:
        """ Test the preparation of the unitary matrix from a `qickit.primitives.Operator`.
        """

    @abstractmethod
    def test_apply_unitary_ndarray(self) -> None:
        """ Test the application of the unitary matrix from a numpy array.
        """

    @abstractmethod
    def test_apply_unitary_operator(self) -> None:
        """ Test the application of the unitary matrix from a `qickit.primitives.Operator`.
        """

    @abstractmethod
    def test_apply_unitary_invalid_input(self) -> None:
        """ Test the application of the unitary matrix with an invalid input.
        """

    @abstractmethod
    def test_apply_unitary_invalid_qubit_indices(self) -> None:
        """ Test the application of the unitary matrix with invalid qubit indices.
        """

    @abstractmethod
    def test_apply_unitary_invalid_qubit_indices_out_of_range(self) -> None:
        """ Test the application of the unitary matrix with qubit indices out of range.
        """