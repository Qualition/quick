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

__all__ = [
    "test_eq",
    "test_len",
    "test_str",
    "test_repr"
]

import pytest
from typing import Type

from qickit.circuit import Circuit

from tests.circuit import CIRCUIT_FRAMEWORKS


@pytest.mark.parametrize("circuit_frameworks", [CIRCUIT_FRAMEWORKS])
def test_eq(circuit_frameworks: list[Type[Circuit]]) -> None:
    """ Test the `__eq__` dunder method.
    """
    circuits = [circuit_framework(2) for circuit_framework in circuit_frameworks]

    # Define the Bell state
    for circuit in circuits:
        circuit.H(0)
        circuit.CX(0, 1)

    # Test the equality of the circuits
    for circuit_1, circuit_2 in zip(circuits[0:-1:], circuits[1::]):
        assert circuit_1 == circuit_2

@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
def test_len(circuit_framework: Type[Circuit]) -> None:
    """ Test the `__len__` dunder method.
    """
    # Define the circuits
    circuit = circuit_framework(2)

    # Define the Bell state
    circuit.H(0)
    circuit.CX(0, 1)

    # Test the length of the circuit
    assert len(circuit) == 2

@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
def test_str(circuit_framework: Type[Circuit]) -> None:
    """ Test the `__str__` dunder method.
    """
    # Define the circuits
    circuit = circuit_framework(2)

    # Define the Bell state
    circuit.H(0)
    circuit.CX(0, 1)

    # Test the string representation of the circuits
    assert str(circuit) == f"{circuit_framework.__name__}(num_qubits=2)"

@pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
def test_repr(circuit_framework: Type[Circuit]) -> None:
    """ Test the `__repr__` dunder method.
    """
    # Define the circuits
    circuit = circuit_framework(2)

    # Define the Bell state
    circuit.H(0)
    circuit.CX(0, 1)

    # Test the string representation of the circuits
    circuit_checker = (
        f"{circuit_framework.__name__}(num_qubits=2, "
        "circuit_log=[{'gate': 'H', 'qubit_indices': 0}, "
        "{'gate': 'CX', 'control_index': 0, 'target_index': 1}])"
    )
    assert repr(circuit) == circuit_checker