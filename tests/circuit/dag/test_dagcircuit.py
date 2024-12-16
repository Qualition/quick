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

__all__ = ["TestDAGCircuit"]

from qickit.circuit.dag import DAGNode, DAGCircuit


class TestDAGCircuit:
    """ `tests.circuit.dag.TestDAGCircuit` is the tester class for `qickit.circuit.dag.DAGCircuit`.
    """
    def test_init(self) -> None:
        """ Test the initialization of a `DAGCircuit` object.
        """
        circuit = DAGCircuit(2)
        assert circuit.num_qubits == 2
        assert circuit.qubits == {"Q0": DAGNode("Q0"), "Q1": DAGNode("Q1")}

    def test_add_operation(self) -> None:
        """ Test the `add_operation` method of a `DAGCircuit` object.
        """
        circuit = DAGCircuit(2)
        circuit.add_operation({"gate": "H", "qubit_indices": 0})
        circuit.add_operation({"gate": "CX", "control_index": 0, "target_index": 1})

        assert circuit.qubits["Q0"].children["Q0"].name == "H"
        assert circuit.qubits["Q0"].children["Q0"].children["Q0"].name == "CX"
        assert circuit.qubits["Q1"].children["Q1"].name == "CX"
        assert circuit.qubits["Q1"].children["Q1"].children == {}
        assert circuit.qubits["Q0"].children["Q0"].children["Q0"].children == {}

    def test_get_depth(self) -> None:
        """ Test the `get_depth` method of a `DAGCircuit` object.
        """
        circuit = DAGCircuit(2)
        circuit.add_operation({"gate": "H", "qubit_indices": 0})
        circuit.add_operation({"gate": "CX", "control_index": 0, "target_index": 1})

        assert circuit.get_depth() == 2

        circuit.add_operation({"gate": "H", "qubit_indices": 1})

        assert circuit.get_depth() == 3

        circuit.add_operation({"gate": "X", "qubit_indices": 0})

        assert circuit.get_depth() == 3