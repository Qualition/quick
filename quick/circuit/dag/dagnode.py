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

from __future__ import annotations

__all__ = ["DAGNode"]

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable


@dataclass
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
    `parents` : set[quick.circuit.dag.DAGNode], optional, default=set()
        A set of parent nodes.
    `children` : set[quick.circuit.dag.DAGNode], optional, default=set()
        A set of children nodes.

    Usage
    -----
    >>> node1 = DAGNode("Node 1")
    """
    name: Hashable = None
    parents: set[DAGNode] = field(default_factory=set)
    children: set[DAGNode] = field(default_factory=set)

    def _invalidate_depth(self) -> None:
        """ Invalidate the cached depth of the node.

        Notes
        -----
        This method is used to invalidate the cached depth of the node
        and all its parents. This is useful when the depth of the node
        needs to be recalculated.

        We delete the `_depth` attribute to indicate that the depth is no
        longer cached. We then call this method recursively on all parent
        nodes to invalidate their depths.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node1._invalidate_depth()
        """
        del self._depth

        for parent in self.parents:
            parent._invalidate_depth()

    def to(
            self,
            child: DAGNode
        ) -> None:
        """ Add a child node to this node.

        Notes
        -----
        This method is used to add a child node to this node. This is
        done by adding the child to the `children` attribute of this
        node and adding this node to the `parents` attribute of the
        child node.

        Additionally, if the depth of this node is cached, we invalidate
        the cached depth.

        Parameters
        ----------
        `child` : quick.circuit.dag.DAGNode
            The next node to add.

        Raises
        ------
        TypeError
            - If `next` is not an instance of `quick.circuit.dag.DAGNode`.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1.to(node2)
        """
        if not isinstance(child, DAGNode):
            raise TypeError(
                "The `next` node must be an instance of DAGNode. "
                f"Received {type(child)} instead."
            )

        self.children.add(child)
        child.parents.add(self)

        if hasattr(self, "_depth"):
            self._invalidate_depth()

    @property
    def depth(self) -> int:
        """ Get the depth of the node.

        Notes
        -----
        The depth of a node is the maximum depth of its children plus one.
        If the node has no children, the depth is zero. The depth is cached
        to avoid recalculating it multiple times.

        Returns
        -------
        int
            The depth of the node.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1.to(node2)
        >>> node1.depth
        """
        if not self.children:
            self._depth = 0

        if not hasattr(self, "_depth"):
            self._calculate_depth()

        return self._depth

    def _calculate_depth(self) -> None:
        """ Calculate the depth of the node.

        Notes
        -----
        We perform the following steps to calculate the depth of the node:
        1. Define the subgraph rooted at the current node.
        2. Calculate the in-degree of each node in the subgraph.
        3. Perform Kahn's algorithm to topologically sort the nodes.
        4. Calculate the depth of each node by working from the leaves.
        """
        in_degrees: dict[DAGNode, int] = {}
        stack: list[DAGNode] = [self]
        queue: deque[DAGNode] = deque([self])

        # Add all nodes in the subgraph rooted at self to the in-degrees dictionary
        # except nodes that already have their depth calculated
        while stack:
            node = stack.pop()
            if node in in_degrees or hasattr(node, "_depth"):
                continue

            # Set the in-degree of the node to zero
            # (this value is arbitrary, we just need a placeholder)
            in_degrees[node] = 0
            for child in node.children:
                stack.append(child)

        # Calculate the in-degree of each node by counting the number of parents
        # in the subgraph rooted at self
        for node in in_degrees:
            in_degrees[node] = sum(parent in in_degrees for parent in node.parents)

        # Perform Kahn's algorithm to topologically sort the nodes
        while queue:
            node = queue.popleft()
            stack.append(node)

            for child in node.children:
                if child not in in_degrees:
                    continue

                in_degrees[child] -= 1
                if in_degrees[child] == 0:
                    queue.append(child)

        # Calculate the depth of each node by working from the leaves and
        # moving up the graph until we reach root
        for node in reversed(stack):
            if not node.children:
                node._depth = 0
            else:
                node._depth = max(child._depth for child in node.children) + 1

    def _generate_paths(
            self,
            node: 'DAGNode',
            path: tuple[Hashable, ...],
            paths: set[tuple[Hashable, ...]]
        ) -> None:
        """ Helper method to recursively generate paths from the current node.

        Parameters
        ----------
        `node` : DAGNode
            The current node being visited.
        `path` : tuple[Hashable, ...]
            The current path being constructed.
        `paths` : set[tuple[Hashable, ...]]
            A set to store all the paths.
        """
        if not node.children:
            paths.add(path)
            return

        for child in node.children:
            self._generate_paths(child, path + (child.name,), paths)

    def generate_paths(self) -> set[tuple[Hashable, ...]]:
        """Generate all paths from this node to the children nodes.

        Returns
        -------
        `paths` : set[tuple[Hashable, ...]]
            A set of tuples representing the paths from this node
            to the children nodes.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node2 = DAGNode("Node 2")
        >>> node1.to(node2)
        >>> node1.generate_paths()
        """
        paths: set[tuple[Hashable, ...]] = set()
        self._generate_paths(self, (self.name,), paths)
        return paths

    def __hash__(self) -> int:
        """ Get the hash of the node.

        Returns
        -------
        int
            The hash of the node.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> hash(node1)
        """
        return hash(id(self))

    def __eq__(
            self,
            other: object
        ) -> bool:
        """ Check if two nodes are equal.

        Parameters
        ----------
        `other` : object
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
        if not isinstance(other, DAGNode):
            return False

        return self.name == other.name and self.children == other.children

    def __repr__(self) -> str:
        """ Get the string representation of the node.

        Returns
        -------
        str
            The string representation of the node.

        Usage
        -----
        >>> node1 = DAGNode("Node 1")
        >>> node1
        """
        return f"{self.name} -> {self.children}" if self.children else f"{self.name}"