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

""" Mottonen approach for preparing quantum states using uniformly controlled rotations.
"""

from __future__ import annotations

__all__ = ["Mottonen"]

import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Bra, Ket
from qickit.synthesis.statepreparation import StatePreparation
from qickit.synthesis.statepreparation.statepreparation_utils import (
    compute_alpha_y, compute_alpha_z, compute_control_indices, compute_m
)


class Mottonen(StatePreparation):
    """ `qickit.synthesis.statepreparation.Mottonen` is the class for preparing quantum states
    using the Möttönen method.

    Notes
    ----------
    The Möttönen method uses uniformly controlled rotations about the y-axis and z-axis to prepare the state.
    This method is based on the paper "Transformation of quantum states using uniformly controlled rotations",
    and scales exponentially with the number of qubits in terms of circuit depth.

    For more information on Möttönen method:
    - Möttönen, Vartiainen, Bergholm, Salomaa.
    Transformation of quantum states using uniformly controlled rotations (2004)
    https://arxiv.org/abs/quant-ph/0407010

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
    >>> state_preparer = Mottonen(output_framework=QiskitCircuit)
    """
    def prepare_state(
            self,
            state: NDArray[np.complex128] | Bra | Ket,
            compression_percentage: float=0.0,
            index_type: Literal["row", "snake"]="row"
        ) -> Circuit:

        if not isinstance(state, (np.ndarray, Bra, Ket)):
            try:
                state = np.array(state)
            except Exception as e:
                raise TypeError(f"The state must be a numpy array or a Bra/Ket object. Received {type(state)} instead.") from e

        match state:
            case np.ndarray():
                state = Ket(state)
            case Bra():
                state = state.to_ket()

        # Order indexing (if required)
        if index_type != "row":
            state.change_indexing(index_type)

        # Compress the statevector values
        state.compress(compression_percentage)

        # Define the number of qubits needed to represent the state
        num_qubits = state.num_qubits

        state = state.data.flatten() # type: ignore

        # Construct Mottonen circuit
        circuit: Circuit = self.output_framework(num_qubits)

        def k_controlled_uniform_rotation(
                circuit: Circuit,
                alpha_k: list[float],
                control_qubits: list[int],
                target_qubit: int,
                rotation_gate: Literal["RY", "RZ"]
            ) -> None:
            """ Apply a k-controlled rotation about the y-axis.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                The quantum circuit.
            `alpha_k` : NDArray[np.float64]
                The array of alphas.
            `control_qubits` : list[int]
                The list of control qubits.
            `target_qubit` : int
                The target qubit.
            `rotation_gate` : Literal["RY", "RZ"]
                The type of gate to be applied.
            """
            gate_mapping = {
                "RY": lambda: circuit.RY,
                "RZ": lambda: circuit.RZ
            }

            k = len(control_qubits)
            thetas = compute_m(k) @ alpha_k
            ctl = compute_control_indices(k)

            for i in range(2**k):
                gate_mapping[rotation_gate]()(thetas[i], target_qubit)
                if k > 0:
                    circuit.CX(control_qubits[k - 1 - ctl[i]], target_qubit)

        # Define the magnitude and phase of the state
        magnitude = np.abs(state) # type: ignore
        phase = np.angle(state) # type: ignore

        # Prepare the state
        for k in range(num_qubits):
            alpha_k = [compute_alpha_y(magnitude, num_qubits - k, j) for j in range(2**k)]
            k_controlled_uniform_rotation(circuit, alpha_k, list(range(k)), k, "RY")

        if not np.all(phase == 0):
            for k in range(num_qubits):
                alpha_k = [compute_alpha_z(phase, num_qubits - k, j) for j in range(2**k)]
                if len(alpha_k) > 0:
                    k_controlled_uniform_rotation(circuit, alpha_k, list(range(k)), k, "RZ")

        # Apply the global phase
        global_phase = sum(phase / len(state))
        circuit.GlobalPhase(global_phase)

        circuit.vertical_reverse()

        if isinstance(state, Bra):
            circuit.horizontal_reverse()

        return circuit