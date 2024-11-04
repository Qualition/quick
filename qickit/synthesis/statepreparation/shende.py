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

""" Shende approach for preparing quantum states using multiplexed RY and RZ gates.
"""

from __future__ import annotations

__all__ = ["Shende"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, SupportsIndex, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Bra, Ket
from qickit.synthesis.statepreparation import StatePreparation
from qickit.synthesis.statepreparation.statepreparation_utils import rotations_to_disentangle


class Shende(StatePreparation):
    """ `qickit.synthesis.statepreparation.Shende` is the class for preparing quantum states
    using the Shende method.

    Notes
    -----
    The Shende method is a recursive method that uses multiplexed RY and RZ gates to prepare the state.
    This method is based on the paper "Synthesis of Quantum Logic Circuits", and scales exponentially
    with the number of qubits in terms of circuit depth.

    For more information on Shende method:
    - Shende, Bullock, Markov.
    Synthesis of Quantum Logic Circuits (2004)
    [https://arxiv.org/abs/quant-ph/0406176v5]

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `qickit.circuit.Circuit`.

    Usage
    -----
    >>> state_preparer = Shende(output_framework=QiskitCircuit)
    """
    def apply_state(
            self,
            circuit: Circuit,
            state: NDArray[np.complex128] | Bra | Ket,
            qubit_indices: int | Sequence[int],
            compression_percentage: float=0.0,
            index_type: Literal["row", "snake"]="row"
        ) -> Circuit:

        if not isinstance(state, (np.ndarray, Bra, Ket)):
            try:
                state = np.array(state).astype(complex)
            except (ValueError, TypeError):
                raise TypeError(f"The state must be a numpy array or a Bra/Ket object. Received {type(state)} instead.")

        match state:
            case np.ndarray():
                state = Ket(state)
            case Bra():
                state = state.to_ket()

        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        if not all(isinstance(qubit_index, SupportsIndex) for qubit_index in qubit_indices):
            raise TypeError("All qubit indices must be integers.")

        if not len(qubit_indices) == state.num_qubits:
            raise ValueError("The number of qubit indices must match the number of qubits in the state.")

        # Order indexing (if required)
        if index_type != "row":
            state.change_indexing(index_type)

        # Compress the statevector values
        state.compress(compression_percentage)

        # Define the number of qubits needed to represent the state
        num_qubits = state.num_qubits

        statevector = state.data.flatten() # type: ignore

        def multiplexor(
                list_of_angles: list[float],
                rotation_gate: Literal["RY", "RZ"],
                last_cnot=True
            ) -> Circuit:
            """ Create the multiplexor circuit, where each instruction itself
            has a decomposition based on smaller multiplexors.

            The LSB is the multiplexor "data" and the other bits are multiplexor "select".

            Parameters
            ----------
            `list_of_angles` : list[float]
                The list of rotation angles.
            `rotation_gate` : Literal["RY", "RZ"]
                The type of gate to be applied.
            `last_cnot` : bool
                Whether to apply the last CNOT gate or not.

            Returns
            -------
            `circuit` : qickit.circuit.Circuit
                The multiplexor circuit.
            """
            # Calculate the number of angles
            num_angles = len(list_of_angles)

            # Define the number of qubits for the local state
            local_num_qubits = int(np.log2(num_angles)) + 1

            # Define the multiplexor circuit
            circuit: Circuit = self.output_framework(local_num_qubits)

            # Define the gate mapping
            gate_mapping = {
                "RY": lambda: circuit.RY,
                "RZ": lambda: circuit.RZ
            }

            # Define least significant bit (LSB) and most significant bit (MSB)
            lsb, msb = 0, local_num_qubits - 1

            # Define the base case for the recursion
            if local_num_qubits == 1:
                gate_mapping[rotation_gate]()(list_of_angles[0], 0)
                return circuit

            # Calculate angle weights
            angle_weight = np.kron([
                [0.5, 0.5],
                [0.5, -0.5]
            ], np.identity(2 ** (local_num_qubits - 2)))

            # Calculate the dot product of the angle weights and the list of angles
            # to get the combo angles
            list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

            # Define the first half multiplexed circuit
            multiplex_1 = multiplexor(list_of_angles[0 : (num_angles // 2)], rotation_gate=rotation_gate, last_cnot=False)
            circuit.add(multiplex_1, list(range(local_num_qubits-1)))

            # Apply CX to flip the LSB qubit
            circuit.CX(msb, lsb)

            # Optimize the circuit by cancelling adjacent CXs
            # (by leaving out last CX and reversing (NOT inverting) the
            # second lower-level multiplex)
            multiplex_2 = multiplexor(list_of_angles[(num_angles // 2) :], rotation_gate=rotation_gate, last_cnot=False)

            if num_angles > 1:
                multiplex_2.horizontal_reverse(adjoint=False)

            circuit.add(multiplex_2, list(range(local_num_qubits-1)))

            # Leave out the last cnot
            if last_cnot:
                circuit.CX(msb, lsb)

            return circuit

        def gates_to_uncompute(
                params: NDArray[np.complex128],
                num_qubits: int
            ) -> Circuit:
            """ Create a circuit with gates that take the desired vector to zero.

            Parameters
            ----------
            `params` : NDArray[np.complex128]
                The list of parameters.
            `num_qubits` : int
                The number of qubits.

            Returns
            -------
            `circuit` : qickit.circuit.Circuit
                The circuit that applies the corresponding gates to uncompute the state.
            """
            # Define the circuit
            circuit: Circuit = self.output_framework(num_qubits)

            # Begin the peeling loop, and disentangle one-by-one from LSB to MSB
            remaining_param = params

            for i in range(num_qubits):
                # Define which rotations must be done to disentangle the LSB
                # qubit (we peel away one qubit at a time)
                (remaining_param, thetas, phis) = rotations_to_disentangle(remaining_param)

                # Perform the required rotations to decouple the LSB qubit (so that
                # it can be "factored" out, leaving a shorter amplitude vector to peel away)
                add_last_cnot = True

                if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                    add_last_cnot = False

                if np.linalg.norm(phis) != 0:
                    rz_mult = multiplexor(list_of_angles=phis, rotation_gate="RZ", last_cnot=add_last_cnot)
                    circuit.add(rz_mult, list(range(i, num_qubits)))

                if np.linalg.norm(thetas) != 0:
                    ry_mult = multiplexor(list_of_angles=thetas, rotation_gate="RY", last_cnot=add_last_cnot)
                    ry_mult.horizontal_reverse(adjoint=False)
                    circuit.add(ry_mult, list(range(i, num_qubits)))

            global_phase_angle = -np.angle(sum(remaining_param))
            circuit.GlobalPhase(float(global_phase_angle))

            return circuit

        # Define the disentangling circuit
        disentangling_circuit = gates_to_uncompute(statevector, num_qubits) # type: ignore

        # We must reverse the circuit to prepare the state,
        # as the circuit is uncomputing from the target state
        # to the zero state
        disentangling_circuit.horizontal_reverse()

        # Add the disentangling circuit to the initial circuit
        circuit.add(disentangling_circuit, qubit_indices)

        return circuit