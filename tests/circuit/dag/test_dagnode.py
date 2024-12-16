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

__all__ = ["TestDAGNode"]

import pytest

from qickit.circuit.dag import DAGNode


class TestDAGNode:
    """ `tests.circuit.dag.TestDAGNode` is the tester class for `qickit.circuit.dag.DAGNode`.
    """
    def test_init(self) -> None:
        """ Test the initialization of a `DAGNode` object.
        """
        dagnode = DAGNode("test_node")
        assert dagnode.name == "test_node"
        assert dagnode.children == {}

    def test_to(self) -> None:
        """ Test the `to` method of a `DAGNode` object.
        """
        dagnode1 = DAGNode("node1")
        dagnode2 = DAGNode("node2")
        dagnode3 = DAGNode("node3")

        dagnode1.to(dagnode2)
        dagnode1.to(dagnode3)

        assert dagnode1.children["node1"] == dagnode2
        assert dagnode2.children["node1"] == dagnode3
        assert dagnode3.children == {}

    def test_generate_paths(self) -> None:
        """ Test the `generate_paths` method of a `DAGNode` object.
        """
        q0 = DAGNode("Q0")
        q1 = DAGNode("Q1")
        H = DAGNode("H")
        CX = DAGNode("CX")
        X = DAGNode("X")
        Y = DAGNode("Y")

        q0.to(H)
        q0.to(CX)
        q1.to(CX)
        q1.to(X)
        q0.to(Y)

        q0_paths = q0.generate_paths()
        q1_paths = q1.generate_paths()

        assert q0_paths == [["Q0", "H", "CX", "X"], ["Q0", "H", "CX", "Y"]]
        assert q1_paths == [["Q1", "CX", "X"], ["Q1", "CX", "Y"]]

    def test_get_depth(self) -> None:
        """ Test the `get_depth` method of a `DAGNode` object.
        """
        q0 = DAGNode("Q0")
        q1 = DAGNode("Q1")
        H = DAGNode("H")
        CX = DAGNode("CX")
        X = DAGNode("X")
        Y = DAGNode("Y")

        q0.to(H)
        q0.to(CX)
        q1.to(CX)
        q1.to(X)
        q0.to(Y)

        assert q0.get_depth() == 3
        assert q1.get_depth() == 2

    def test_to_invalid(self) -> None:
        """ Test the `to` method of a `DAGNode` object with an invalid argument.
        """
        dagnode1 = DAGNode("node1")
        dagnode2 = "node2"

        with pytest.raises(TypeError):
            dagnode1.to(dagnode2) # type: ignore

    def test_str(self) -> None:
        """ Test the string representation of a `DAGNode` object.
        """
        dagnode = DAGNode("test_node")
        assert str(dagnode) == "Name: test_node, Children: [{}]"