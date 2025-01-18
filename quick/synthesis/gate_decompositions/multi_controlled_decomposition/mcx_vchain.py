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

""" MCX V-Chain decomposition module for multi-controlled gates.

This implementation is based on Qiskit's implementation.
https://github.com/Qiskit/qiskit/blob/stable/0.46/qiskit/circuit/library/standard_gates/multi_control_rotation_gates.py
"""

from __future__ import annotations

__all__ = ["MCXVChain"]

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from quick.circuit import Circuit
from quick.synthesis.gate_decompositions.multi_controlled_decomposition.mcx_utils import CCX, C3X


class MCXVChain:
    """ `quick.synthesis.gate_decompositions.MCXVChain` class is used to perform V-chain decomposition of MCX gates.

    Notes
    -----
    This decomposition is to be used for decomposing multi-controlled gates with
    only U3 and CX gates. This decomposition is only usable for 4 or more control
    qubits. If the number of control qubits is less than 4, the decomposition will
    default to the standard MCX decomposition for CX, CCX and C3X gates.

    It can synthesize a multi-controlled X gate with k controls using k - 2 dirty
    ancillary qubits producing a circuit with 2 * k - 1 qubits and at most
    8 * k - 6 CX gates, by Iten et. al.

    This implementation is based on the following paper:
    - Iten, Colbeck, Kukuljan, Home, Christandl.
    Quantum circuits for isometries (2016).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    Usage
    -----
    >>> mcx_vchain = MCXVChain()
    """
    @staticmethod
    def get_num_ancillas(num_controls) -> int:
        """ Get the number of ancilla qubits required for the V-chain decomposition.

        Parameters
        ----------
        `num_controls` : int
            Number of control qubits.

        Returns
        -------
        `num_ancillas` : int
            Number of ancilla qubits required for the V-chain decomposition.

        Usage
        -----
        >>> num_ancillas = MCXVChain.get_num_ancillas(4)
        """
        return max(0, num_controls - 2)

    def define_decomposition(
            self,
            num_controls: int,
            output_framework: Type[Circuit]
        ) -> Circuit:
        """ Define the V-chain decomposition of the MCX gate.

        Parameters
        ----------
        `num_controls` : int
            Number of control qubits for the MCX gate.
        `output_framework` : Type[quick.circuit.Circuit]
            The circuit framework to be used for the decomposition.

        Returns
        -------
        `circuit` : quick.circuit.Circuit
            The circuit decomposition of the MCX gate using V-Chain.

        Usage
        -----
        >>> mcx_vchain.define_decomposition(4, QiskitCircuit)
        """
        circuit = output_framework(num_controls + self.get_num_ancillas(num_controls) + 1)
        qubits = list(range(circuit.num_qubits))

        self.apply_decomposition(
            circuit,
            qubits[:num_controls],
            qubits[num_controls],
            qubits[num_controls + 1:]
        )

        return circuit

    def apply_decomposition(
            self,
            circuit: Circuit,
            control_indices: int | list[int],
            target_index: int,
            ancilla_indices: int | list[int] | None = None
        ) -> None:
        """ Apply the V-chain decomposition of the MCX gate.

        Parameters
        ----------
        `circuit` : quick.circuit.Circuit
            The circuit framework to be used for the decomposition.
        `control_indices` : int | list[int]
            The control qubits for the MCX gate.
        `target_index` : int
            The target qubit for the MCX gate.
        `ancilla_indices` : int | list[int], optional, default: []
            The ancilla qubits for the MCX gate. If not provided, it will be empty.
            This may raise an error if ancilla qubits are needed.

        Raises
        ------
        ValueError
            - If the number of qubits in the circuit is not enough for the decomposition.
            - If the number of ancilla qubits provided is not equal to the number of ancilla qubits required.

        Usage
        -----
        >>> mcx_vchain.apply_decomposition(circuit, [0, 1, 2, 3], 4, [5, 6])
        """
        if ancilla_indices is None:
            ancilla_indices = []

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        ancilla_indices = [ancilla_indices] if isinstance(ancilla_indices, int) else ancilla_indices

        num_control_qubits = len(control_indices)
        num_ancillas = len(ancilla_indices) if ancilla_indices else self.get_num_ancillas(num_control_qubits)

        if circuit.num_qubits < num_control_qubits + num_ancillas + 1:
            raise ValueError(
                "The number of qubits in the circuit is not enough for MCX V-Chain decomposition. "
                f"Received {circuit.num_qubits} qubits. "
                f"Required {num_control_qubits + num_ancillas + 1} qubits."
            )

        if len(ancilla_indices) != num_ancillas:
            raise ValueError(
                "The number of ancilla qubits provided is not equal to the number of ancilla qubits required. "
                f"Received {len(ancilla_indices)} ancilla qubits. "
                f"Required {num_ancillas} ancilla qubits."
            )

        # The V-chain decomposition for the MCX gate only works for 4+ control qubits
        if num_control_qubits == 1:
            circuit.CX(control_indices[0], target_index)
            return
        elif num_control_qubits == 2:
            CCX(circuit, control_indices, target_index)
            return
        elif num_control_qubits == 3:
            C3X(circuit, control_indices, target_index)
            return

        targets = [target_index] + ancilla_indices[::-1]

        # Perform V-Chain decomposition of the MCX gate
        for _ in range(2):
            for i in range(num_control_qubits):
                if i < num_control_qubits - 2:
                    if targets[i] != target_index:
                        circuit.H(targets[i])
                        circuit.T(targets[i])
                        circuit.CX(control_indices[num_control_qubits - i - 1], targets[i])
                        circuit.Tdg(targets[i])
                        circuit.CX(ancilla_indices[num_ancillas - i - 1], targets[i])
                    else:
                        controls = [
                            control_indices[num_control_qubits - i - 1],
                            ancilla_indices[num_ancillas - i - 1],
                        ]
                        CCX(circuit, [controls[0], controls[1]], targets[i])
                else:
                    circuit.H(targets[i])
                    circuit.T(targets[i])
                    circuit.CX(control_indices[num_control_qubits - i - 2], targets[i])
                    circuit.Tdg(targets[i])
                    circuit.CX(control_indices[num_control_qubits - i - 1], targets[i])
                    circuit.T(targets[i])
                    circuit.CX(control_indices[num_control_qubits - i - 2], targets[i])
                    circuit.Tdg(targets[i])
                    circuit.H(targets[i])
                    break

            for i in range(num_ancillas - 1):
                circuit.CX(ancilla_indices[i], ancilla_indices[i + 1])
                circuit.T(ancilla_indices[i + 1])
                circuit.CX(control_indices[2 + i], ancilla_indices[i + 1])
                circuit.Tdg(ancilla_indices[i + 1])
                circuit.H(ancilla_indices[i + 1])