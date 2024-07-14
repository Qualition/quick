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

__all__ = ["StatePreparation", "Mottonen", "Shende"]

from abc import ABC, abstractmethod
import numpy as np
from typing import Literal, overload, Type

from qickit.circuit import Circuit
from qickit.primitives import Bra, Ket
from qickit.synthesis.statepreparation_utils import (theta, M, ind, alpha_y, alpha_z,
                                                     rotations_to_disentangle)
from qickit.types import Collection, Scalar


class StatePreparation(ABC):
    """ `qickit.synthesis.statepreparation.StatePreparation` is the class for preparing quantum states.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Usage
    -----
    >>> state_preparer = StatePreparation(output_framework=QiskitCircuit)
    """
    def __init__(self,
                 output_framework: Type[Circuit]) -> None:
        """ Initalize a State Preparation instance.
        """
        self.output_framework = output_framework

    @overload
    @abstractmethod
    def prepare_state(self,
                      state: Collection[Scalar],
                      compression_percentage: float=0.0,
                      index_type: str = "row") -> Circuit:
        """ Prepare the quantum state.

        Parameters
        ----------
        `state` : qickit.types.Collection[qickit.types.Scalar]
            The quantum state to prepare.
        `compression_percentage` : float, optional, default=0.0
            Number between 0 an 100, where 0 is no compression and 100 all statevector values are 0.
        `index_type` : str, optional, default="row"
            The indexing type for the data.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit that prepares the state.
        """

    @overload
    @abstractmethod
    def prepare_state(self,
                      state: Bra,
                      compression_percentage: float=0.0,
                      index_type: str = "row") -> Circuit:
        """ Prepare the quantum state.

        Parameters
        ----------
        `state` : qickit.primitives.Bra
            The quantum state to prepare.
        `compression_percentage` : float, optional, default=0.0
            Number between 0 an 100, where 0 is no compression and 100 all statevector values are 0.
        `index_type` : str, optional, default="row"
            The indexing type for the data.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit that prepares the state.
        """

    @overload
    @abstractmethod
    def prepare_state(self,
                      state: Ket,
                      compression_percentage: float = 0.0,
                      index_type: str = "row") -> Circuit:
        """ Prepare the quantum state.

        Parameters
        ----------
        `state` : qickit.primitives.Ket
            The quantum state to prepare.
        `compression_percentage` : float, optional, default=0.0
            Number between 0 an 100, where 0 is no compression and 100 all statevector values are 0.
        `index_type` : str, optional, default="row"
            The indexing type for the data.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit that prepares the state.
        """


class Mottonen(StatePreparation):
    """ `qickit.synthesis.statepreparation.Mottonen` is the class for preparing quantum states using the Mottonen method.
    """
    def prepare_state(self,
                      state: Collection[Scalar] | Bra | Ket,
                      compression_percentage: float=0.0,
                      index_type: str="row") -> Circuit:
        if isinstance(state, Collection):
            state = Ket(state)

        # Order indexing (if required)
        if index_type != "row":
            state.change_indexing(index_type)

        # Compress the statevector values
        state.compress(compression_percentage)

        # Define the number of qubits needed to represent the state
        num_qubits = state.num_qubits

        state = state.data.flatten()

        # Construct Mottonen circuit
        circuit: Circuit = self.output_framework(num_qubits)

        def k_controlled_uniform_rotation_y(circuit: Circuit,
                                            alpha_k: list[float],
                                            control_qubits: list[int],
                                            target_qubit: int) -> None:
            """ Apply a k-controlled rotation about the y-axis.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                The quantum circuit.
            `alpha_k` : list[float]
                The array of alphas.
            `control_qubits` : list[int]
                The list of control qubits.
            `target_qubit` : int
                The target qubit.
            """
            # Define the number of qubits
            k = len(control_qubits)
            # Calulate the number of parameters
            n = 2**k
            # Define thetas
            thetas = theta(M(k), alpha_k)

            # If there are no control qubits, apply a RY gate to the target qubit
            if k == 0:
                circuit.RY(thetas[0], target_qubit)
            else :
                # Define the control index required for the amplitude encoding circuit
                control_index = ind(k)
                # Iterate over the parameters, and apply the RY gate and the CX gate
                for i in range(n) :
                    circuit.RY(thetas[i], target_qubit)
                    circuit.CX(control_qubits[k-1-control_index[i]], target_qubit)

        def k_controlled_uniform_rotation_z(circuit: Circuit,
                                            alpha_k: list[float],
                                            control_qubits: list[int],
                                            target_qubit: int) -> None:
            """ Apply a k-controlled rotation about the z-axis.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                The quantum circuit.
            `alpha_k` : list[float]
                The array of alphas.
            `control_qubits` : list[int]
                The list of control qubits.
            `target_qubit` : int
                The target qubit.
            """
            # Define the number of qubits
            k = len(control_qubits)
            # Calulate the number of parameters
            n = 2**k
            # Define thetas
            thetas = theta(M(k), alpha_k)

            # If there are no control qubits, apply a RZ gate to the target qubit
            if k == 0:
                circuit.RZ(thetas[0], target_qubit)
            else :
                # Define the control index required for the amplitude encoding circuit
                control_index = ind(k)
                # Iterate over the parameters, and apply the RZ gate and the CX gate
                for i in range(n) :
                    circuit.RZ(thetas[i], target_qubit)
                    circuit.CX(control_qubits[k-1-control_index[i]], target_qubit)

        # Define the magnitude and phase of the state
        magnitude = np.abs(state)
        phase = np.angle(state)

        # Prepare the state
        for k in range(num_qubits):
            alpha_k = [alpha_y(magnitude, num_qubits - k, j) for j in range(2 ** k)]
            k_controlled_uniform_rotation_y(circuit, alpha_k, list(range(k)), k)

        if not np.all(phase == 0):
            for k in range(num_qubits):
                alpha_k = [alpha_z(phase, num_qubits - k, j) for j in range(2 ** k)]
                if len(alpha_k) > 0:
                    k_controlled_uniform_rotation_z(circuit, alpha_k, list(range(k)), k)

        # Apply the global phase
        global_phase = sum(np.angle(state) / len(state))
        circuit.GlobalPhase(global_phase)

        circuit.vertical_reverse()

        return circuit


class Shende(StatePreparation):
    """ `qickit.synthesis.statepreparation.Shende` is the class for preparing quantum states
    using the Shende method.

    The Shende method is a recursive method that uses multiplexed RY and RZ gates to prepare the state.
    This method is based on the paper "Synthesis of Quantum Logic Circuits" [1], and scales exponentially
    with the number of qubits in terms of circuit depth.

    References
    ----------
    [1] Shende, Bullock, Markov. Synthesis of Quantum Logic Circuits (2004)
    [https://arxiv.org/abs/quant-ph/0406176v5]
    """
    def prepare_state(self,
                      state: Collection[Scalar] | Bra | Ket,
                      compression_percentage: float=0.0,
                      index_type: str="row") -> Circuit:
        if isinstance(state, Collection):
            state = Ket(state)

        # Order indexing (if required)
        if index_type != "row":
            state.change_indexing(index_type)

        # Compress the statevector values
        state.compress(compression_percentage)

        # Define the number of qubits needed to represent the state
        num_qubits = state.num_qubits

        state = state.data.flatten()

        # Construct Shende circuit
        circuit: Circuit = self.output_framework(num_qubits)

        def multiplexor(list_of_angles: list[float],
                        gate: Literal["RY", "RZ"],
                        last_cnot=True) -> Circuit:
            """ Create the multiplexor circuit, where each instruction itself
            has a decomposition based on smaller multiplexors.

            The LSB is the multiplexor "data" and the other bits are multiplexor "select".

            Parameters
            ----------
            `list_of_angles` : list[float]
                The list of rotation angles.
            `gate` : Literal["RY", "RZ"]
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
                "RY": circuit.RY,
                "RZ": circuit.RZ
            }

            # Define least significant bit (LSB) and most significant bit (MSB)
            lsb, msb = 0, local_num_qubits - 1

            # Define the base case for the recursion
            if local_num_qubits == 1:
                gate_mapping[gate](list_of_angles[0], 0)
                return circuit

            # Calculate angle weights
            angle_weight = np.kron([[0.5, 0.5],
                                    [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))

            # Calculate the dot product of the angle weights and the list of angles
            # to get the combo angles
            list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

            # Define the first half multiplexed circuit
            multiplex_1 = multiplexor(list_of_angles[0 : (num_angles // 2)], gate=gate, last_cnot=False)
            circuit.add(multiplex_1, list(range(local_num_qubits-1)))

            # Apply CX to flip the LSB qubit
            circuit.CX(msb, lsb)

            # Optimize the circuit by cancelling adjacent CXs
            # (by leaving out last CX and reversing (NOT inverting) the
            # second lower-level multiplex)
            multiplex_2 = multiplexor(list_of_angles[(num_angles // 2) :], gate=gate, last_cnot=False)
            if num_angles > 1:
                multiplex_2.horizontal_reverse(adjoint=False)
                circuit.add(multiplex_2, list(range(local_num_qubits-1)))
            else:
                circuit.add(multiplex_2, list(range(local_num_qubits-1)))

            # Leave out the last cnot
            if last_cnot:
                circuit.CX(msb, lsb)

            return circuit

        def gates_to_uncompute(params: Collection[complex],
                               num_qubits: int) -> Circuit:
            """ Create a circuit with gates that take the desired vector to zero.

            Parameters
            ----------
            `params` : Collection[complex]
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
                    rz_mult = multiplexor(list_of_angles=phis, gate="RZ", last_cnot=add_last_cnot)
                    circuit.add(rz_mult, list(range(i, num_qubits)))

                if np.linalg.norm(thetas) != 0:
                    ry_mult = multiplexor(list_of_angles=thetas, gate="RY", last_cnot=add_last_cnot)
                    ry_mult.horizontal_reverse(adjoint=False)
                    circuit.add(ry_mult, list(range(i, num_qubits)))

            circuit.GlobalPhase(-np.angle(sum(remaining_param)))
            return circuit

        # Define the disentangling circuit
        disentangling_circuit = gates_to_uncompute(state, num_qubits)
        # Apply a horizontal reverse (adjoint)
        disentangling_circuit.horizontal_reverse()
        # Add the disentangling circuit to the initial circuit
        circuit.add(disentangling_circuit, list(range(num_qubits)))

        return circuit