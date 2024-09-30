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

""" Shende's Shannon decomposition for preparing quantum unitary operators
using multiplexed RY and RZ gates.
"""

from __future__ import annotations

__all__ = ["ShannonDecomposition"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cossin # type: ignore
from typing import Literal, SupportsIndex, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Operator
from qickit.synthesis.statepreparation.statepreparation_utils import gray_code
from qickit.synthesis.unitarypreparation import UnitaryPreparation
from qickit.synthesis.unitarypreparation.unitarypreparation_utils import deconstruct_single_qubit_matrix_into_angles


class ShannonDecomposition(UnitaryPreparation):
    """ `qickit.ShannonDecomposition` is the class for preparing quantum operators
    using Shannon decomposition.

    Notes
    -----
    Shende's Shannon decomposition uses multiplexed RY and RZ gates to prepare the unitary
    operator. This method scales exponentially with the number of qubits in terms of circuit
    depth.

    ```
       ┌───┐               ┌───┐     ┌───┐     ┌───┐
      ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
       │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
     /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
       └───┘          └───┘     └───┘     └───┘     └───┘
    ```

    For more information on Shannon decomposition:
    - Shende, Bullock, Markov.
    Synthesis of Quantum Logic Circuits (2006)
    https://arxiv.org/abs/quant-ph/0406176

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
    """
    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        def quantum_shannon_decomposition(
                circuit: Circuit,
                qubits: list[int],
                unitary: NDArray[np.complex128]
            ) -> None:
            """ Decompose n-qubit unitary into CX/RY/RZ/CX gates, preserving global phase.

            Using cosine-sine decomposition, the unitary matrix is decomposed into a series of
            single-qubit rotations and CX gates. The most significant qubit is then decomposed
            into a series of RY rotations and CX gates, and the process is repeated recursively
            until the unitary is fully decomposed.

            ```
              ┌───┐               ┌───┐
            ──┤   ├──      ────□──┤ Ry├──□───
              │ U │    =     ┌─┴─┐└─┬─┘┌─┴─┐
            /─┤   ├──      /─┤ U ├──□──┤ V ├─
              └───┘          └───┘     └───┘
            ```

            The algorithm is described in Shende et al.:
            Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
            https://arxiv.org/abs/quant-ph/0406176

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                Quantum circuit to append operations to.
            `qubits` : list[int]
                List of qubits in order of significance.
            `unitary` : NDArray[np.complex128]
                N-qubit unitary matrix to be decomposed.

            Raises
            ------
            ValueError
                - If the u matrix is non-unitary
                - If the u matrix is not of shape (2^n,2^n)
            """
            n = unitary.shape[0]

            if n == 2:
                single_qubit_decomposition(circuit, qubits, unitary) # type: ignore
                return

            # Perform a cosine-sine (linalg) decomposition on the unitary
            (u1, u2), theta, (v1, v2) = cossin(unitary, separate=True, p=n/2, q=n/2)

            # Apply the decomposition of multiplexed v1/v2 part
            msb_demuxer(circuit, qubits, v1, v2)

            # Apply the multiplexed RY gate
            multiplexed_cossin(circuit, qubits, theta, "RY")

            # Apply the decomposition of multiplexed u1/u2 part
            msb_demuxer(circuit, qubits, u1, u2)

        def single_qubit_decomposition(
                circuit: Circuit,
                qubit: int,
                unitary: NDArray[np.complex128]) -> None:
            """ Decompose single-qubit gate to RY and RZ rotations and global phase.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                Quantum circuit to append operations to.
            `qubit` : int
                Qubit on which to apply operations.
            `unitary` : NDArray[np.complex128]
                2x2 unitary matrix representing 1-qubit gate to be decomposed
            """
            # Perform ZYZ decomposition
            phi_0, phi_1, phi_2 = deconstruct_single_qubit_matrix_into_angles(unitary)

            # Calculate the global phase picked up by the decomposition
            phase = np.angle(unitary[0, 0] / (np.exp(-1j * phi_0 / 2) * np.cos(phi_1 / 2)))

            # Apply the ZYZ decomposition
            circuit.RZ(phi_0, qubit)
            circuit.RY(phi_1, qubit)
            circuit.Phase(phi_2, qubit)

            # Apply global phase e^{i*pi*phase} picked up during the decomposition
            circuit.GlobalPhase(phase)

        def msb_demuxer(
                circuit: Circuit,
                demux_qubits: list[int],
                unitary_1: NDArray[np.complex128],
                unitary_2: NDArray[np.complex128]
            ) -> None:
            """ Decompose a multiplexor defined by a pair of unitary matrices operating on the same subspace.

            That is, decompose

            ```
              ctrl     ────□────
                        ┌──┴──┐
              target  /─┤     ├─
                        └─────┘
            ```

            represented by the block diagonal matrix

            ```
                ┏         ┓
                ┃ U1      ┃
                ┃      U2 ┃
                ┗         ┛
            ```

            to

            ```
                             ┌───┐
              ctrl    ───────┤ Rz├──────
                        ┌───┐└─┬─┘┌───┐
              target  /─┤ W ├──□──┤ V ├─
                        └───┘     └───┘
            ```

            by means of simultaneous unitary diagonalization.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                Quantum circuit to append operations to.
            `demux_qubits` : list[int]
                Subset of total qubits involved in this unitary gate.
            `unitary_1` : NDArray[np.complex128]
                Upper-left quadrant of total unitary to be decomposed (see diagram).
            `unitary_2` : NDArray[np.complex128]
                Lower-right quadrant of total unitary to be decomposed (see diagram).
            """
            # Compute the product of `unitary_1` and the conjugate transpose of `unitary_2`
            u = unitary_1 @ unitary_2.conj().T

            # Perform eigenvalue decomposition to find the eigenvalues and eigenvectors of u
            # This step is crucial because it allows us to express the unitary transformation
            # in terms of its eigenvalues and eigenvectors, which simplifies further calculations
            d_squared, V = np.linalg.eig(u)

            # Take the square root of the eigenvalues to obtain the singular values
            # This is necessary because the singular values provide a more convenient form
            # for constructing the diagonal matrix D, which is used in the final decomposition
            d = np.sqrt(d_squared)

            # Create a diagonal matrix D from the singular values
            # The diagonal matrix D is used to scale the eigenvectors appropriately in the final step
            D = np.diag(d)

            # Compute the matrix W using D, the conjugate transpose of V, and `unitary_2`
            # This step combines the scaled eigenvectors with the original unitary matrix to
            # achieve the desired decomposition
            W = D @ V.conj().T @ unitary_2

            # Apply the left gate
            quantum_shannon_decomposition(circuit, demux_qubits[1:], W)

            # Apply the RZ multiplexed gate
            multiplexed_cossin(circuit, demux_qubits, -np.angle(d), "RZ")

            # Apply the right gate
            quantum_shannon_decomposition(circuit, demux_qubits[1:], V)

        def multiplexed_cossin(
                circuit: Circuit,
                cossin_qubits: list[int],
                angles: NDArray[np.floating],
                rotation_gate: Literal["RY", "RZ"]
            ) -> None:
            """ Perform a multiplexed rotation over all qubits in this unitary matrix.
            This function uses RY and RZ multiplexing for quantum shannon decomposition.

            Parameters
            ----------
            `circuit` : qickit.circuit.Circuit
                Quantum circuit to append operations to.
            `cossin_qubits` : list[int]
                Subset of total qubits involved in this unitary gate.
            `angles` : list[float]
                List of angles to be multiplexed over for the given type of rotation.
            `rotation_gate` : Literal["RY", "RZ"]
                Rotation function used for this multiplexing implementation (RY or RZ).
            """
            # Most significant qubit is main qubit with rotation function applied
            # All other qubits are control qubits
            main_qubit = cossin_qubits[0]
            control_qubits = cossin_qubits[1:]

            for j in range(len(angles)):
                # The rotation includes a factor of -1 for each bit in the Gray Code
                # if the position of that bit is also 1
                # The number of factors of -1 is counted using the 1s in the
                # binary representation of the (gray(j) & i)
                # Here, i gives the index for the angle, and
                # j is the iteration of the decomposition
                rotation = sum(
                    -angle if bin(gray_code(j) & i).count("1") % 2 else angle
                    for i, angle in enumerate(angles)
                )

                # Divide by a factor of 2 for each additional select qubit
                # This is due to the halving in the decomposition applied recursively
                rotation = rotation * 2 / len(angles)

                # The XOR of the this gray code with the next will give the 1 for the bit
                # corresponding to the CX select, else 0
                select_string = gray_code(j) ^ gray_code(j + 1)

                # Find the index number where the bit is 1
                select_qubit = next(i for i in range(len(angles)) if (select_string >> i & 1))

                # Negate the value, since we must index starting at most significant qubit
                # Also the final value will overflow, and it should be the MSB,
                # therefore introduce max function
                select_qubit = max(-select_qubit - 1, -len(control_qubits))

                # Define the gate mapping
                gate_mapping = {
                    "RY": lambda: circuit.RY,
                    "RZ": lambda: circuit.RZ
                }

                # Add a rotation on the main qubit
                gate_mapping[rotation_gate]()(rotation, main_qubit)

                # Add a CX main qubit controlled by the select qubit
                circuit.CX(control_qubits[select_qubit], main_qubit)

        # Apply the Shannon decomposition to the circuit
        quantum_shannon_decomposition(circuit, qubit_indices[::-1], unitary.data) # type: ignore

        return circuit