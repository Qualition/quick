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

""" Iten approach for preparing quantum states using isometries.

This implementation is based on the implementation in the Qiskit library.
https://github.com/Qiskit/qiskit/blob/stable/0.46/qiskit/circuit/library/generalized_gates/isometry.py
"""

from __future__ import annotations

__all__ = ["Isometry"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, SupportsIndex, TYPE_CHECKING

if TYPE_CHECKING:
    from quick.circuit import Circuit
from quick.circuit.circuit_utils import extract_single_qubits_and_diagonal
from quick.primitives import Bra, Ket
from quick.synthesis.statepreparation import StatePreparation
from quick.synthesis.statepreparation.statepreparation_utils import (
    a,
    b,
    k_s,
    get_binary_rep_as_list,
    get_qubits_by_label,
    reverse_qubit_state,
    apply_multi_controlled_gate,
    apply_diagonal_gate,
    apply_diagonal_gate_to_diag,
    disentangling_single_qubit_gates,
    ucg_is_identity_up_to_global_phase,
    merge_ucgate_and_diag,
    apply_ucg
)


class Isometry(StatePreparation):
    """ `quick.synthesis.statepreparation.Isometry` is the class for preparing quantum states
    using the Iten method.

    Notes
    -----
    The Iten method uses isometries to reduce the depth by a factor of 2 compared to Shende and Möttönen.
    This method is based on the paper "Quantum circuits of isometries" by Iten et al.,
    and scales exponentially with the number of qubits in terms of circuit depth.

    For more information on Iten method:
    [1] Iten, Colbeck, Kukuljan, Home, Christandl.
    Quantum circuits for isometries (2016).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    Parameters
    ----------
    `output_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `quick.circuit.Circuit`.

    Usage
    -----
    >>> state_preparer = Isometry(output_framework=QiskitCircuit)
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

        if isinstance(state, np.ndarray):
            state = Ket(state)
        elif isinstance(state, Bra):
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
        qubits = list(range(num_qubits))

        state = state.data.flatten() # type: ignore
        state = state.reshape(state.shape[0], 1) # type: ignore

        # Construct Isometry circuit
        isometry_circuit: Circuit = self.output_framework(num_qubits)

        def apply_multiplexor_up_to_diagonal(
                circuit: Circuit,
                single_qubit_gates: list[NDArray[np.complex128]],
                control_labels: list[int],
                target_label: int
            ) -> NDArray[np.complex128]:
            """ Apply a multiplexed gate to the circuit up to diagonal.

            Parameters
            ----------
            `circuit` : quick.circuit.Circuit
                The quantum circuit.
            `single_qubit_gates` : list[NDArray[np.complex128]]
                The single qubit gates.
            `control_labels` : list[int]
                The control qubit labels.
            `target_label` : int
                The target qubit label.

            Returns
            -------
            `diagonal` : NDArray[np.complex128]
                The diagonal gate.
            """
            control_indices = list(reversed(
                get_qubits_by_label(control_labels, qubits, num_qubits)
            ))
            target_index = get_qubits_by_label([target_label], qubits, num_qubits)[0]

            circuit.Multiplexor(
                control_indices=control_indices,
                target_index=target_index,
                single_qubit_gates=single_qubit_gates,
                up_to_diagonal=True
            )

            # Extract the diagonal gate from the decomposition of the multiplexed gate
            (_, diagonal) = extract_single_qubits_and_diagonal(single_qubit_gates, len(control_labels) + 1)

            return diagonal

        def disentangle(
                circuit: Circuit,
                diagonal: NDArray[np.complex128],
                remaining_isometry: NDArray[np.complex128],
                s: int
            ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
            """ Disentangle the s-th significant qubit (starting with s=0) into the zero
            or the one state (dependent on column_index).

            Parameters
            ----------
            `circuit` : quick.circuit.Circuit
                The quantum circuit.
            `diagonal` : NDArray[np.complex128]
                The diagonal gate.
            `remaining_isometry` : NDArray[np.complex128]
                The remaining isometry.
            `s` : int
                The significant qubit index.

            Returns
            -------
            `v` : NDArray[np.complex128]
                The remaining isometry.
            `diagonal` : NDArray[np.complex128]
                The diagonal gate.
            """
            v = remaining_isometry

            # An MCG gate is a multiplexed gate with all single qubit gates being identity
            # besides the last one which is the matrix that reverses the qubit state
            index_1 = 2 * a(0, s + 1) * 2**s + b(0, s + 1)
            index_2 = (2 * a(0, s + 1) + 1) * 2**s + b(0, s + 1)
            target_label = num_qubits - s - 1

            # Check if the MCG gate is needed
            if (
                k_s(0, s) == 0
                and b(0, s + 1) != 0
                and np.abs(v[index_2, 0]) > 1e-10
            ):
                gate = reverse_qubit_state(
                    np.array([v[index_1, 0], v[index_2, 0]]), 0
                )

                control_labels = [
                    i
                    for i, x in enumerate(get_binary_rep_as_list(0, num_qubits))
                    if x == 1 and i != target_label
                ]

                single_qubit_gates = [np.eye(2).astype(complex) for _ in control_labels]
                single_qubit_gates.append(gate)

                diagonal_mcg = apply_multiplexor_up_to_diagonal(
                    circuit, single_qubit_gates, control_labels, target_label
                )

                # Apply the MCG to the remaining isometry
                v = apply_multi_controlled_gate(v, control_labels, target_label, gate)

                # Correct for the implementation up to diagonal
                diag_mcg_inverse = np.conj(diagonal_mcg).astype(complex, copy=False) # type: ignore
                v = apply_diagonal_gate(
                    v, control_labels + [target_label], diag_mcg_inverse
                )

                # Update the diag according to the applied diagonal gate
                diagonal = apply_diagonal_gate_to_diag(
                    diagonal, control_labels + [target_label], diag_mcg_inverse, num_qubits
                )

            # Find the multiplexed gate, decompose it and apply it to the remaining isometry
            # to disentangle a qubit
            single_qubit_gates = disentangling_single_qubit_gates(v, 0, s, num_qubits) # type: ignore

            # If the multiplexed gate is not identity up to global phase, then we must
            # decompose it and apply it to the remaining isometry
            if not ucg_is_identity_up_to_global_phase(single_qubit_gates):
                control_labels = list(range(target_label))

                diagonal_ucg = apply_multiplexor_up_to_diagonal(
                    circuit, single_qubit_gates, control_labels, target_label
                )

                # Merge the diagonal into the UC gate for efficient application of both together
                diagonal_ucg_inverse = np.conj(diagonal_ucg).astype(complex, copy=False) # type: ignore
                single_qubit_gates = merge_ucgate_and_diag(
                    single_qubit_gates, diagonal_ucg_inverse
                )

                # Apply the UC gate (with the merged diagonal gate) to the remaining isometry
                v = apply_ucg(v, len(control_labels), single_qubit_gates)

                # Update the diagonal according to the applied diagonal gate
                diagonal = apply_diagonal_gate_to_diag(
                    diagonal, control_labels + [target_label], diagonal_ucg_inverse, num_qubits
                )

            return v, diagonal

        # Apply the gates to uncompute the state
        diagonal = np.array([])

        # Iteratively disentangle the qubits of the state
        # from the target state to the zero state
        for i in range(num_qubits):
            state, diagonal = disentangle(
                isometry_circuit, diagonal, state, i
            )

        # We must reverse the circuit to prepare the state,
        # as the circuit is uncomputing from the target state
        # to the zero state
        # If the state is a bra, we will apply a horizontal reverse
        # which will nullify this reverse, thus we will first check
        # if the state is not a bra before applying the reverse
        if not isinstance(state, Bra):
            isometry_circuit.horizontal_reverse()

        # Add the isometry circuit to the initial circuit
        circuit.add(isometry_circuit, qubit_indices)

        return circuit