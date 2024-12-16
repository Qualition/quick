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

__all__ = ["DAGNode"]


class DAGNode:
    """ A node in a directed acyclic graph (DAG).

    Notes
    -----
    Quantum circuits can be represented using DAGs, where each node represents
    a quantum operation and each edge represents a qubit that the operation acts
    on. This class is used to represent a node in a DAG.

    In this implementation we omit using an edge class and instead use a list of
    children to represent the edges. This is because the edge class would only
    contain a reference to the child node, and so we can simplify the implementation
    by storing the children directly in the node.

    Parameters
    ----------
    `name` : str
        The name of the node.

    Attributes
    ----------
    `name` : str
        The name of the node.
    `children` : dict[str, qickit.circuit.dag.DAGNode]
        A dictionary of children nodes, where the keysare the names
        of the children.

    Usage
    -----
    >>> node1 = DAGNode("Node 1")
    """
    def __init__(
            self,
            name: str
        ) -> None:
        """ Initialize a DAGNode.
        """
        self.name = name
        self.children: dict[str, 'DAGNode'] = {}

    def to(
            self,
            next: 'DAGNode'
        ) -> None:
        """ Add a child node to this node.

        Parameters
        ----------
        `next` : qickit.circuit.dag.DAGNode
            The next node to add.

        Raises
        ------
        TypeError
            If `next` is not an instance of `DAGNode`.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1.to(node2)
        """
        if not isinstance(next, DAGNode):
            raise TypeError("The next node must be an instance of DAGNode.")

        if not self.children:
            self.children = {self.name: next}

        else:
            children = self.children[self.name].children

            while children:
                if self.name not in children:
                    break

                children = children[self.name].children

            children[self.name] = next

    def _generate_paths(
            self,
            node: 'DAGNode',
            path: list[str],
            paths: list[list[str]]
        ) -> None:
        """ Helper method to recursively generate paths from the current node.

        Parameters
        ----------
        `node` : DAGNode
            The current node being visited.
        `path` : list[str]
            The current path being constructed.
        `paths` : list[list[str]]
            A list to store all the paths.
        """
        if not node.children:
            paths.append(path)
            return

        for child in node.children.values():
            self._generate_paths(child, path + [child.name], paths)

    def generate_paths(self) -> list[list[str]]:
        """Generate all paths from this node to the children nodes.

        Returns
        -------
        `paths` : list[list[str]]
            A list of strings representing the paths from this node
            to the children nodes.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1.to(node2)
        >>> node1.generate_paths()
        """
        paths: list[list[str]] = []
        self._generate_paths(self, [self.name], paths)
        return paths

    def get_depth(self) -> int:
        """ Get the depth of the node.

        Returns
        -------
        int
            The depth of the node.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1.to(node2)
        >>> node1.get_depth()
        """
        return max(len(path) for path in self.generate_paths()) - 1

    def __eq__(self, value: object) -> bool:
        """ Check if this node is equal to another node.

        Parameters
        ----------
        `value` : object
            The object to compare to.

        Returns
        -------
        bool
            True if the nodes are equal, False otherwise.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1 == node2
        """
        if not isinstance(value, DAGNode):
            return False

        return self.name == value.name and self.children == value.children

    def __repr__(self) -> str:
        """ Get a string representation of the node.

        Returns
        -------
        str
            A string representation of the node.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> repr(node1)
        """
        return f"Name: {self.name}, Children: [{self.children}]"