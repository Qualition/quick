# Copyright 2023-2024 Qualition Computing LLC.
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

__all__ = ["DAGCircuit"]

from quick.circuit.dag import DAGNode


class DAGCircuit:
    """ A directed acyclic graph (DAG) representation of a quantum circuit.

    Notes
    -----
    Quantum circuits can be represented using DAGs, where each node represents
    a quantum operation and each edge represents a qubit that the operation acts
    on. This class is used to represent a quantum circuit as a DAG.

    Parameters
    ----------
    `num_qubits` : int
        The number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        The number of qubits in the circuit.
    `qubits` : dict[str, quick.circuit.dag.DAGNode]
        A dictionary mapping qubit indices to nodes in the DAG.
    `stack` : dict[str, list[quick.circuit.dag.DAGNode]]
        A dictionary mapping qubit indices to a stack of nodes in the DAG.

    Usage
    -----
    >>> dag = DAGCircuit(2)
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:
        """ Initialize a `quick.circuit.dag.DAGCircuit` instance.
        """
        self.num_qubits = num_qubits
        self.qubits = {f"Q{i}": DAGNode(f"Q{i}") for i in range(num_qubits)}
        self.stack = {f"Q{i}": [self.qubits[f"Q{i}"]] for i in range(num_qubits)}

    def add_operation(
            self,
            operation: dict
        ) -> None:
        """ Add an operation to the circuit.

        Parameters
        ----------
        `operation` : dict
            A dictionary representing the operation to add to
            the circuit. This dictionary is the same as what
            is added to `quick.circuit.Circuit.circuit_log`.

        Usage
        -----
        >>> dag.add_operation({
        ...     "name": "H",
        ...     "qubit_indices": 0
        ... })
        """
        from quick.circuit.circuit import ALL_QUBIT_KEYS

        gate_node = DAGNode(operation["gate"])
        qubit_indices = []

        # Add qubits from any valid qubit key to the
        # qubit indices
        for key in operation:
            if key in ALL_QUBIT_KEYS:
                if isinstance(operation[key], int):
                    qubit_indices.append(operation[key])
                elif isinstance(operation[key], list):
                    qubit_indices.extend(operation[key])

        for qubit in qubit_indices:
            self.stack[f"Q{qubit}"].append(gate_node)
            self.stack[f"Q{qubit}"][-2].to(gate_node)

    def get_depth(self) -> int:
        """ Get the depth of the circuit.

        Returns
        -------
        int
            The depth of the circuit.

        Usage
        -----
        >>> dag.get_depth()
        """
        return max(qubit.depth for qubit in self.qubits.values())

    def __repr__(self) -> str:
        """ Get the string representation of the circuit.

        Returns
        -------
        str
            The string representation of the circuit.

        Usage
        -----
        >>> dag = DAGCircuit(2)
        >>> dag
        """
        return "\n".join([f"{qubit}: {stack[0]}" for qubit, stack in self.stack.items()])