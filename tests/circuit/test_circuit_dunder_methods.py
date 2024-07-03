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

__all__ = ["test_eq",
           "test_len",
           "test_str",
           "test_repr"]

from qickit.circuit import CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit


def test_eq() -> None:
    """ Test the `__eq__` dunder method.
    """
    # Define the circuits
    cirq_circuit = CirqCircuit(2)
    pennylane_circuit = PennylaneCircuit(2)
    qiskit_circuit = QiskitCircuit(2)
    tket_circuit = TKETCircuit(2)

    # Define the Bell state
    cirq_circuit.H(0)
    cirq_circuit.CX(0, 1)

    pennylane_circuit.H(0)
    pennylane_circuit.CX(0, 1)

    qiskit_circuit.H(0)
    qiskit_circuit.CX(0, 1)

    tket_circuit.H(0)
    tket_circuit.CX(0, 1)

    # Test the equality of the circuits
    assert cirq_circuit == pennylane_circuit
    assert cirq_circuit == qiskit_circuit
    assert cirq_circuit == tket_circuit

def test_len() -> None:
    """ Test the `__len__` dunder method.
    """
    # Define the circuits
    cirq_circuit = CirqCircuit(2)
    pennylane_circuit = PennylaneCircuit(2)
    qiskit_circuit = QiskitCircuit(2)
    tket_circuit = TKETCircuit(2)

    # Define the Bell state
    cirq_circuit.H(0)
    cirq_circuit.CX(0, 1)

    pennylane_circuit.H(0)
    pennylane_circuit.CX(0, 1)

    qiskit_circuit.H(0)
    qiskit_circuit.CX(0, 1)

    tket_circuit.H(0)
    tket_circuit.CX(0, 1)

    # Test the length of the circuits
    assert len(cirq_circuit) == 2
    assert len(pennylane_circuit) == 2
    assert len(qiskit_circuit) == 2
    assert len(tket_circuit) == 2

def test_str() -> None:
    """ Test the `__str__` dunder method.
    """
    # Define the circuits
    cirq_circuit = CirqCircuit(2)
    pennylane_circuit = PennylaneCircuit(2)
    qiskit_circuit = QiskitCircuit(2)
    tket_circuit = TKETCircuit(2)

    # Define the Bell state
    cirq_circuit.H(0)
    cirq_circuit.CX(0, 1)

    pennylane_circuit.H(0)
    pennylane_circuit.CX(0, 1)

    qiskit_circuit.H(0)
    qiskit_circuit.CX(0, 1)

    tket_circuit.H(0)
    tket_circuit.CX(0, 1)

    # Test the string representation of the circuits
    check = "[{'gate': 'H', 'qubit_indices': 0}, {'gate': 'CX', 'control_index': 0, 'target_index': 1}]"
    assert str(cirq_circuit) == check
    assert str(pennylane_circuit) == check
    assert str(qiskit_circuit) == check
    assert str(tket_circuit) == check

def test_repr() -> None:
    """ Test the `__repr__` dunder method.
    """
    # Define the circuits
    cirq_circuit = CirqCircuit(2)
    pennylane_circuit = PennylaneCircuit(2)
    qiskit_circuit = QiskitCircuit(2)
    tket_circuit = TKETCircuit(2)

    # Define the Bell state
    cirq_circuit.H(0)
    cirq_circuit.CX(0, 1)

    pennylane_circuit.H(0)
    pennylane_circuit.CX(0, 1)

    qiskit_circuit.H(0)
    qiskit_circuit.CX(0, 1)

    tket_circuit.H(0)
    tket_circuit.CX(0, 1)

    # Test the string representation of the circuits
    check = "Circuit(num_qubits=2)"
    assert repr(cirq_circuit) == check
    assert repr(pennylane_circuit) == check
    assert repr(qiskit_circuit) == check
    assert repr(tket_circuit) == check