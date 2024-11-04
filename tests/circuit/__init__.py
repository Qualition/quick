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

__all__ = [
    "CIRCUIT_FRAMEWORKS",
    "test_from_circuit",
    "TestControlled",
    "Template",
    "FrameworkTemplate",
    "TestCirqCircuit",
    "TestFromCirq",
    "TestPennylaneCircuit",
    "TestQiskitCircuit",
    "TestFromQiskit",
    "TestTKETCircuit",
    "TestFromTKET",
    "TestUniformlyControlledGates",
    "test_qft_no_swap_no_inverse",
    "test_qft_swap_no_inverse",
    "test_qft_no_swap_inverse",
    "test_qft_swap_inverse",
    "test_eq",
    "test_len",
    "test_str",
    "test_repr"
]

from qickit.circuit import CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit

# The quantum circuit frameworks
CIRCUIT_FRAMEWORKS = [
    CirqCircuit,
    PennylaneCircuit,
    QiskitCircuit,
    TKETCircuit
]

from tests.circuit.test_circuit_convert import test_from_circuit
from tests.circuit.test_circuit_to_controlled import TestControlled
from tests.circuit.test_circuit import Template
from tests.circuit.test_from_framework import FrameworkTemplate
from tests.circuit.test_cirq_circuit import TestCirqCircuit
from tests.circuit.test_circuit_from_cirq import TestFromCirq
from tests.circuit.test_pennylane_circuit import TestPennylaneCircuit
from tests.circuit.test_qiskit_circuit import TestQiskitCircuit
from tests.circuit.test_circuit_from_qiskit import TestFromQiskit
from tests.circuit.test_tket_circuit import TestTKETCircuit
from tests.circuit.test_circuit_from_tket import TestFromTKET
from tests.circuit.test_circuit_dunder_methods import test_eq, test_len, test_str, test_repr
from tests.circuit.test_qft_circuit import (
    test_qft_no_swap_no_inverse,
    test_qft_swap_no_inverse,
    test_qft_no_swap_inverse,
    test_qft_swap_inverse
)