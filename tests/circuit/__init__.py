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

__all__ = [
    "CIRCUIT_FRAMEWORKS",
    "TestCircuitBase",
    "TestControlled",
    "TestControlState",
    "Template",
    "TestCirqCircuit",
    "TestPennylaneCircuit",
    "TestQiskitCircuit",
    "TestQuimbCircuit",
    "TestTKETCircuit",
    "TestUniformlyControlledGates",
    "TestQFTCircuit"
]

from quick.circuit import (
    CirqCircuit, PennylaneCircuit, QiskitCircuit, QuimbCircuit, TKETCircuit
)

# The quantum circuit frameworks
CIRCUIT_FRAMEWORKS = [
    CirqCircuit,
    PennylaneCircuit,
    QiskitCircuit,
    QuimbCircuit,
    TKETCircuit
]

from tests.circuit.test_circuit_base import TestCircuitBase
from tests.circuit.test_circuit_to_controlled import TestControlled
from tests.circuit.test_control_state import TestControlState
from tests.circuit.test_circuit import Template
from tests.circuit.test_cirq_circuit import TestCirqCircuit
from tests.circuit.test_pennylane_circuit import TestPennylaneCircuit
from tests.circuit.test_qiskit_circuit import TestQiskitCircuit
from tests.circuit.test_quimb_circuit import TestQuimbCircuit
from tests.circuit.test_tket_circuit import TestTKETCircuit
from tests.circuit.test_qft_circuit import TestQFTCircuit
from tests.circuit.test_uniformly_controlled_gates import TestUniformlyControlledGates