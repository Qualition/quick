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

__all__ = ['StatePreparation', 'Mottonen', 'Shende']

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Type

# Import `qickit.data.Data`
from qickit.data import Data

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit

# Import `qickit.synthesis.statepreparation_utils.py` methods
from qickit.synthesis.statepreparation_utils import (theta, M, ind, alpha_y,
                                                     rotations_to_disentangle)

# Import `qickit.types.Collection`
from qickit.types import Collection


class StatePreparation(ABC):
    """ `qickit.synthesis.statepreparation.StatePreparation` is the class for preparing quantum states.

    Parameters
    ----------
    `circuit_framework` : qickit.circuit.Circuit
        The quantum circuit framework.

    Attributes
    ----------
    `circuit_framework` : qickit.circuit.Circuit
        The quantum circuit framework.
    """
    def __init__(self,
                 circuit_framework: Type[Circuit]) -> None:
        """ Initalize a State Preparation instance.
        """
        # Define the QC framework
        self.circuit_framework = circuit_framework

    @abstractmethod
    def prepare_state(self,
                      state: NDArray[np.complex128] | Data,
                      compression_percentage: float = 0.0,
                      index_type: str = 'row') -> Circuit:
        """ Prepare the quantum state.

        Parameters
        ----------
        `state` : NDArray[np.complex128] | qickit.data.Data
            The quantum state to prepare.
        `compression_percentage` : float
            Number between 0 an 100, where 0 is no compression and 100 is no image.
        `index_type` : str
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
                      state: NDArray[np.complex128] | Data,
                      compression_percentage: float = 0.0,
                      index_type: str = 'row') -> Circuit:
        # Define a qickit.Data instance
        if not isinstance(state, Data):
            state = Data(state)

        # Order indexing (if required)
        if index_type != 'row':
            state.change_indexing(index_type)

        # Convert the state to a quantum state
        state.to_quantumstate()

        # Compress the data
        state.compress(compression_percentage)

        # Define the number of qubits (n qubits, where n = Log2(N), where N is the dimension of the vector)
        num_qubits = state.num_qubits

        # Flatten the data in case it is not 1-dimensional
        state = state.data.flatten()

        # Construct Mottonen circuit
        circuit: Circuit = self.circuit_framework(num_qubits, num_qubits)

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
            # If there are control qubits
            else :
                # Define the control index required for the amplitude encoding circuit
                control_index = ind(k)
                # Iterate over the parameters
                for i in range(n) :
                    # Apply RY to the target qubit
                    circuit.RY(thetas[i], target_qubit)
                    # Apply CX between the control qubits and the target qubit
                    circuit.CX(control_qubits[k-1-control_index[i]], target_qubit)

        # Iterate over all qubits
        for k in range(num_qubits):
            # Define Alpha k
            alpha_k = [alpha_y(state, num_qubits - k, j) for j in range(2 ** k)]
            # Apply the k controleld rotation about the y-axis
            k_controlled_uniform_rotation_y(circuit, alpha_k, list(range(k)), k)

        circuit.vertical_reverse()

        return circuit


class Shende(StatePreparation):
    """ `qickit.synthesis.statepreparation.Shende` is the class for preparing quantum states using the Shende method.
    """
    def prepare_state(self,
                      state: NDArray[np.complex128] | Data,
                      compression_percentage: float = 0.0,
                      index_type: str = 'row') -> Circuit:
        # Define a qickit.Data instance
        if not isinstance(state, Data):
            state = Data(state)

        # Order indexing (if required)
        if index_type != 'row':
            state.change_indexing(index_type)

        # Convert the state to a quantum state
        state.to_quantumstate()

        # Compress the data
        state.compress(compression_percentage)

        # Define the number of qubits (n qubits, where n = Log2(N), where N is the dimension of the vector)
        num_qubits = state.num_qubits

        # Flatten the data in case it is not 1-dimensional
        state = state.data.flatten()

        # Construct Shende circuit
        circuit: Circuit = self.circuit_framework(num_qubits, num_qubits)

        def multiplex_ry(list_of_angles: list[float],
                         last_cnot=True) -> Circuit:
            """ Take a list of angles and return a circuit that applies the corresponding multiplexed Ry gate.

            Parameters
            ----------
            `list_of_angles` : list[float]
                The list of angles.
            `last_cnot` : bool
                Whether to apply the last CNOT gate or not.

            Returns
            -------
            `circuit` : qickit.circuit.Circuit
                The multiplexed Ry gate's circuit.
            """
            # Calculate the number of angles
            num_angles = len(list_of_angles)

            # Define the number of qubits for the local state
            local_num_qubits = int(np.log2(num_angles)) + 1

            # Define the Multiplex circuit
            circuit: Circuit = self.circuit_framework(local_num_qubits, local_num_qubits)

            # Define LSB and MSB
            lsb, msb = 0, local_num_qubits - 1

            # If there is only one qubit, we apply the target gate to that single qubit
            if local_num_qubits == 1:
                circuit.RY(list_of_angles[0], 0)
                return circuit

            # If there are two qubits, we apply the target gate to the first qubit and then apply a CNOT gate
            angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))

            list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

            # Define the first half multiplexed RY circuit
            multiplex_1 = multiplex_ry(list_of_angles[0 : (num_angles // 2)], False)
            circuit.add(multiplex_1, list(range(local_num_qubits-1)))
            circuit.CX(msb, lsb)

            # Define the second half multiplexed RY circuit
            multiplex_2 = multiplex_ry(list_of_angles[(num_angles // 2) :], False)

            if num_angles > 1:
                multiplex_2.horizontal_reverse(adjoint=False)
                circuit.add(multiplex_2, list(range(local_num_qubits-1)))
            else:
                circuit.add(multiplex_2, list(range(local_num_qubits-1)))

            # If the last CNOT gate is required, we apply it
            if last_cnot:
                circuit.CX(msb, lsb)

            return circuit

        def multiplex_rz(list_of_angles: list[float],
                         last_cnot=True) -> Circuit:
            """ Take a list of angles and return a circuit that applies the corresponding multiplexed RZ gate.

            Parameters
            ----------
            `list_of_angles` : list[float]
                The list of angles.
            `last_cnot` : bool
                Whether to apply the last CNOT gate or not.

            Returns
            -------
            `circuit` : qickit.circuit.Circuit
                The multiplexed Ry gate's circuit.
            """
            # Calculate the number of angles
            num_angles = len(list_of_angles)

            # Define the number of qubits for the local state
            local_num_qubits = int(np.log2(num_angles)) + 1

            # Define the Multiplex circuit
            circuit: Circuit = self.circuit_framework(local_num_qubits, local_num_qubits)

            # Define LSB and MSB
            lsb = 0
            msb = local_num_qubits - 1

            # If there is only one qubit, we apply the target gate to that single qubit
            if local_num_qubits == 1:
                circuit.RZ(list_of_angles[0], 0)
                return circuit

            # If there are two qubits, we apply the target gate to the first qubit and then apply a CNOT gate
            angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (local_num_qubits - 2)))

            list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

            # Define the first half multiplexed RZ gate
            multiplex_1 = multiplex_rz(list_of_angles[0 : (num_angles // 2)], False)
            circuit.add(multiplex_1, list(range(local_num_qubits-1)))
            circuit.CX(msb, lsb)

            # Define the second half multiplexed RZ gate
            multiplex_2 = multiplex_rz(list_of_angles[(num_angles // 2) :], False)

            # If there are more than one angle, apply a horizontal reverse (no adjoint) before adding the multiplex
            if num_angles > 1:
                multiplex_2.horizontal_reverse(adjoint=False)
                circuit.add(multiplex_2, list(range(local_num_qubits-1)))
            else:
                circuit.add(multiplex_2, list(range(local_num_qubits-1)))

            # If the last CNOT gate is required, we apply it
            if last_cnot:
                circuit.CX(msb, lsb)

            return circuit

        def gates_to_uncompute(params: Collection[complex],
                               num_qubits: int) -> Circuit:
            """ Take a list of parameters and return a circuit that applies the corresponding gates
            to uncompute the state.

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
            circuit: Circuit = self.circuit_framework(num_qubits, num_qubits)

            # Define the remaining parameters
            remaining_param = params

            # Iterate over all qubits
            for i in range(num_qubits):
                # Calculate the parameters for disentangling
                (remaining_param, thetas, phis) = rotations_to_disentangle(remaining_param)
                # Set the last cnot to true
                add_last_cnot = True

                # Check if the norm of phis and thetas is not 0
                if np.linalg.norm(phis) != 0 and np.linalg.norm(thetas) != 0:
                    # Set the last cnot to false
                    add_last_cnot = False

                # Check if the norm of phis is not 0
                if np.linalg.norm(phis) != 0:
                    # Apply the multiplexed RZ gate
                    rz_mult = multiplex_rz(phis, last_cnot=add_last_cnot)
                    circuit.add(rz_mult, list(range(i, num_qubits)))

                # Check if the norm of thetas is not 0
                if np.linalg.norm(thetas) != 0:
                    # Apply the multiplexed RY gate
                    ry_mult = multiplex_ry(thetas, last_cnot=add_last_cnot)
                    # Apply a horizontal reverse (not adjoint)
                    ry_mult.horizontal_reverse(adjoint=False)
                    circuit.add(ry_mult, list(range(i, num_qubits)))

            return circuit

        # Define the disentangling circuit
        disentangling_circuit = gates_to_uncompute(state, num_qubits)
        # Apply a horizontal reverse (adjoint)
        disentangling_circuit.horizontal_reverse()
        # Add the disentangling circuit to the initial circuit
        circuit.add(disentangling_circuit, list(range(num_qubits)))

        return circuit