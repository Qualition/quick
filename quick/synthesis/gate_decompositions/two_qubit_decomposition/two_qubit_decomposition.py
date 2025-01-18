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

""" Two-qubit unitary decomposition using the KAK decomposition.

This implementation is based on Jake Lishman's implementation for qiskit-terra:
https://github.com/jakelishman/qiskit-terra/tree/storage/deterministic-weyl-decomposition
"""

from __future__ import annotations

__all__ = ["TwoQubitDecomposition"]

import cmath
from collections.abc import Sequence
import math
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quick.circuit import Circuit
from quick.circuit.gate_matrix import RZ, CX
from quick.primitives import Operator
from quick.synthesis.gate_decompositions.one_qubit_decomposition import OneQubitDecomposition
from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import TwoQubitWeylDecomposition
from quick.synthesis.unitarypreparation import UnitaryPreparation

# Constants
SQRT2 = 1 / np.sqrt(2)

""" Hardcoded basis gates for the KAK decomposition
using the CX gate as the basis.
"""
Q0L = np.array([
    [0.5+0.5j, 0.5-0.5j],
    [-0.5-0.5j, 0.5-0.5j]
], dtype=complex)

Q0R = np.array([
    [-0.5-0.5j, 0.5-0.5j],
    [-0.5-0.5j, -0.5+0.5j]
], dtype=complex)

Q1LA = np.array([
    [0.+0.j, -1-1j],
    [1-1j, 0.+0.j]
], dtype=complex) * SQRT2

Q1LB = np.array([
    [-0.5+0.5j, -0.5-0.5j],
    [0.5-0.5j, -0.5-0.5j]
], dtype=complex)

Q1RA = np.array([
    [1+0.j, 1+0.j],
    [-1+0.j, 1+0.j]
], dtype=complex) * SQRT2

Q1RB = np.array([
    [0.5-0.5j, 0.5+0.5j],
    [-0.5+0.5j, 0.5+0.5j]
], dtype=complex)

Q2L = np.array([
    [-1+1j, 0.+0.j],
    [0.+0.j, -1-1j]
], dtype=complex) * SQRT2

Q2R = np.array([
    [0.+1j, 0.-1j],
    [0.-1j, 0.-1j]
], dtype=complex) * SQRT2

U0L: NDArray[np.complex128] = np.array([
    [-1, 1],
    [-1, -1]
], dtype=complex) * SQRT2

U0R: NDArray[np.complex128] = np.array([
    [-1j, 1j],
    [1j, 1j]
], dtype=complex) * SQRT2

U1L: NDArray[np.complex128] = np.array([
    [-0.5+0.5j, -0.5+0.5j],
    [0.5+0.5j, -0.5-0.5j]
], dtype=complex)

U1RA: NDArray[np.complex128] = np.array([
    [0.5-0.5j, -0.5-0.5j],
    [0.5-0.5j, 0.5+0.5j]
], dtype=complex)

UR1B: NDArray[np.complex128] = np.array([
    [-1, -1j],
    [-1j, -1]
], dtype=complex) * SQRT2

u2la: NDArray[np.complex128] = np.array([
    [0.5+0.5j, 0.5-0.5j],
    [-0.5-0.5j, 0.5-0.5j]
], dtype=complex)

U2LB: NDArray[np.complex128] = np.array([
    [-0.5+0.5j, -0.5-0.5j],
    [0.5-0.5j, -0.5-0.5j]
], dtype=complex)

U2RA: NDArray[np.complex128] = np.array([
    [-0.5+0.5j, 0.5-0.5j],
    [-0.5-0.5j, -0.5-0.5j]
], dtype=complex)

U2RB: NDArray[np.complex128] = np.array([
    [0.5-0.5j, 0.5+0.5j],
    [-0.5+0.5j, 0.5+0.5j]
], dtype=complex)

U3L: NDArray[np.complex128] = np.array([
    [-1+1j, 0+0j],
    [0+0j, -1-1j]
], dtype=complex) * SQRT2

U3R: NDArray[np.complex128] = np.array([
    [1j, -1j],
    [-1j, -1j]
], dtype=complex) * SQRT2


class TwoQubitDecomposition(UnitaryPreparation):
    """ `quick.synthesis.unitarypreparation.TwoQubitDecomposition` is
    the class for decomposing two-qubit unitary matrices into one qubit
    quantum gates and CX gates.

    Notes
    -----
    The decomposition is based on the KAK decomposition, which decomposes a 2-qubit unitary matrix
    into a sequence of three unitary matrices, each of which is a product of one-qubit gates and a
    CX gate.

    The up to diagonal decomposition of two qubit unitaries into the product of a diagonal gate
    and another unitary gate can be represented by two CX gates instead of the usual three.
    This can be used when neighboring gates commute with the diagonal to potentially reduce
    overall CX count.

    To use the up to diagonal decomposition, the `apply_unitary_up_to_diagonal` method can be used.

    For more information on KAK decomposition, refer to the following paper:
    [1] Vidal, Dawson.
    A Universal Quantum Circuit for Two-qubit Transformations with 3 CNOT Gates (2003)
    https://arxiv.org/pdf/quant-ph/0307177

    Parameters
    ----------
    `output_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[quick.circuit.Circuit]
        The quantum circuit framework.
    `one_qubit_decomposition` : quick.synthesis.gate_decompositions.OneQubitDecomposition
        The one-qubit decomposition class.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `quick.circuit.Circuit`.

    Usage
    -----
    >>> two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)
    """
    def __init__(
            self,
            output_framework: type[Circuit],
        ) -> None:
        """ Initialize the decomposer with a 2-qubit gate.
        """
        super().__init__(output_framework)

        self.one_qubit_decomposition = OneQubitDecomposition(output_framework)

    @staticmethod
    def u4_to_su4(u4: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], float]:
        """ Convert a general 4x4 unitary matrix to a SU(4) matrix.

        Parameters
        ----------
        `u4` : NDArray[np.complex128]
            The 4x4 unitary matrix.

        Returns
        -------
        `su4` : NDArray[np.complex128]
            The 4x4 special unitary matrix.
        `phase_factor` : float
            The phase factor.
        """
        phase_factor = np.conj(np.linalg.det(u4) ** (-1 / u4.shape[0]))
        su4: NDArray[np.complex128] = u4 / phase_factor
        return su4, cmath.phase(phase_factor)

    @staticmethod
    def traces(target: TwoQubitWeylDecomposition) -> list[complex]:
        """ Calculate the expected traces $|Tr(U \cdot U_{target}^\dagger)|$
        for different number of basis gates.

        Parameters
        ----------
        `target` : quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl.TwoQubitWeylDecomposition
            The target Weyl decomposition.

        Returns
        -------
        list[complex]
            The expected traces.
        """
        ta, tb, tc = target.a, target.b, target.c

        # b will always be 0 when using CX as KAK basis
        cx_basis_b = 0

        return [
            4
            * complex(
                math.cos(ta) * math.cos(tb) * math.cos(tc),
                math.sin(ta) * math.sin(tb) * math.sin(tc),
            ),
            4
            * complex(
                math.cos(math.pi / 4 - ta) * math.cos(cx_basis_b - tb) * math.cos(tc),
                math.sin(math.pi / 4 - ta) * math.sin(cx_basis_b - tb) * math.sin(tc),
            ),
            4 * math.cos(tc),
            4,
        ]

    @staticmethod
    def real_trace_transform(U: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """ Determine diagonal gate such that

        U3 = D U2

        Where U3 is a general two-qubit gate which takes 3 CX, D is a
        diagonal gate, and U2 is a gate which takes 2 CX.

        Parameters
        ----------
        `U` : NDArray[np.complex128]
            The 4x4 unitary matrix.

        Returns
        -------
        `diagonal` : NDArray[np.complex128]
            The diagonal matrix.
        """
        a1 = (
            -U[1, 3] * U[2, 0] +
            U[1, 2] * U[2, 1] +
            U[1, 1] * U[2, 2] -
            U[1, 0] * U[2, 3]
        )
        a2 = (
            U[0, 3] * U[3, 0] -
            U[0, 2] * U[3, 1] -
            U[0, 1] * U[3, 2] +
            U[0, 0] * U[3, 3]
        )

        # Initialize theta and phi (they can be arbitrary)
        theta = 0
        phi = 0

        psi = np.arctan2(a1.imag + a2.imag, a1.real - a2.real) - phi
        diagonal = np.diag(np.exp(-1j * np.array([theta, phi, psi, -(theta + phi + psi)])))
        return diagonal

    @staticmethod
    def trace_to_fidelity(trace: complex) -> float:
        """ Calculate the average gate fidelity

        ..math::

            \bar{F} = \frac{d + |Tr (U_{target} \cdot U^\dagger)|^2}{d(d+1)}

        Notes
        -----
        The average gate fidelity is a measure of the average fidelity of a quantum gate
        with respect to the ideal gate. The average gate fidelity is defined as the average
        of the gate fidelity over all possible input states.

        For more information on the average gate fidelity, refer to the following paper:
        [1] Horodecki, Horodecki, Horodecki.
        General teleportation channel, singlet fraction, and quasidistillation (1999)
        https://arxiv.org/pdf/quant-ph/9807091

        Parameters
        ----------
        `trace` : complex
            The trace of the unitary matrix.

        Returns
        -------
        float
            The average gate fidelity.
        """
        return (4 + abs(trace) ** 2) / 20

    @staticmethod
    def _decomp0(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """ Decompose target ~Ud(x, y, z) with 0 uses of the basis gate.
        Result Ur has trace:

        ..math::

            |Tr(Ur.U_{target}^\dagger)| = 4|(\cos(x) \cos(y) \cos(z) + i \sin(x) \sin(y) \sin(z)|

        which is optimal for all targets and bases.

        Parameters
        ----------
        `weyl_decomposition` : quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl.TwoQubitWeylDecomposition
            The target Weyl decomposition.

        Returns
        -------
        `U0r` : NDArray[np.complex128]
            The right unitary matrix.
        `U0l` : NDArray[np.complex128]
            The left unitary matrix.
        """
        U0l = weyl_decomposition.K1l.dot(weyl_decomposition.K2l)
        U0r = weyl_decomposition.K1r.dot(weyl_decomposition.K2r)
        return U0r, U0l

    @staticmethod
    def _decomp1(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128]
        ]:
        """ Decompose target ~Ud(x, y, z) with 1 uses of the basis gate ~Ud(a, b, c).
        Result Ur has trace:

        .. math::

            |Tr(Ur.U_{target}^\dagger)| = 4|\cos(x-a) \cos(y-b) \cos(z-c) + i \sin(x-a) \sin(y-b) \sin(z-c)|

        which is optimal for all targets and bases with z==0 or c==0.

        Parameters
        ----------
        `weyl_decomposition` : quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl.TwoQubitWeylDecomposition
            The target Weyl decomposition.

        Returns
        -------
        `U1r` : NDArray[np.complex128]
            The right unitary matrix.
        `U1l` : NDArray[np.complex128]
            The left unitary matrix.
        `U0r` : NDArray[np.complex128]
            The right unitary matrix.
        `U0l` : NDArray[np.complex128]
            The left unitary matrix.
        """
        # Get the CX gate in LSB ordering
        CX.change_mapping("LSB")

        # Use the basis gate as the closest reflection in the Weyl chamber
        basis = TwoQubitWeylDecomposition(CX.matrix)

        U0l = weyl_decomposition.K1l.dot(basis.K1l.T.conj())
        U0r = weyl_decomposition.K1r.dot(basis.K1r.T.conj())
        U1l = basis.K2l.T.conj().dot(weyl_decomposition.K2l)
        U1r = basis.K2r.T.conj().dot(weyl_decomposition.K2r)

        return U1r, U1l, U0r, U0l

    @staticmethod
    def _decomp2_supercontrolled(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128]
        ]:
        """ Decompose target ~Ud(x, y, z) with 2 uses of the basis gate.

        For supercontrolled basis ~Ud(pi/4, b, 0), all b, result Ur has trace

        .. math::

            |Tr(Ur.U_{target}^\dagger)| = 4 \cos(z)

        which is the optimal approximation for basis of CX-class ``~Ud(pi/4, 0, 0)``
        or DCX-class ``~Ud(pi/4, pi/4, 0)`` and any target.

        Notes
        -----
        May be sub-optimal for b!=0 (e.g. there exists exact decomposition for any target using B
        ``B~Ud(pi/4, pi/8, 0)``, but not this decomposition.)
        This is an exact decomposition for supercontrolled basis and target ``~Ud(x, y, 0)``.
        No guarantees for non-supercontrolled basis.

        Parameters
        ----------
        `weyl_decomposition` : quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl.TwoQubitWeylDecomposition
            The target Weyl decomposition.

        Returns
        -------
        `U2r` : NDArray[np.complex128]
            The right unitary matrix.
        `U2l` : NDArray[np.complex128]
            The left unitary matrix.
        `U1r` : NDArray[np.complex128]
            The right unitary matrix.
        `U1l` : NDArray[np.complex128]
            The left unitary matrix.
        `U0r` : NDArray[np.complex128]
            The right unitary matrix.
        `U0l` : NDArray[np.complex128]
            The left unitary matrix.
        """
        U0l = weyl_decomposition.K1l.dot(Q0L)
        U0r = weyl_decomposition.K1r.dot(Q0R)
        U1l = Q1LA.dot(RZ(-2 * float(weyl_decomposition.a)).matrix).dot(Q1LB)
        U1r = Q1RA.dot(RZ(2 * float(weyl_decomposition.b)).matrix).dot(Q1RB)
        U2l = Q2L.dot(weyl_decomposition.K2l)
        U2r = Q2R.dot(weyl_decomposition.K2r)

        return U2r, U2l, U1r, U1l, U0r, U0l

    @staticmethod
    def _decomp3_supercontrolled(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128]
        ]:
        """ Decompose a 2-qubit unitary matrix into a sequence of one-qubit gates and CX gates.
        The decomposition uses three CX gates.

        Notes
        -----
        The decomposition is based on the KAK decomposition, which decomposes a 2-qubit unitary matrix
        into a sequence of three unitary matrices, each of which is a product of one-qubit gates and a
        CX gate.

        For more information on KAK decomposition, refer to the following paper:
        - Vidal, Dawson.
        A Universal Quantum Circuit for Two-qubit Transformations with 3 CNOT Gates (2003)
        https://arxiv.org/pdf/quant-ph/0307177

        Parameters
        ----------
        `weyl_decomposition` : quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl.TwoQubitWeylDecomposition
            The Weyl decomposition of a 2-qubit unitary matrix.

        Returns
        -------
        `U3r` : NDArray[np.complex128]
            The right unitary matrix.
        `U3l` : NDArray[np.complex128]
            The left unitary matrix.
        `U2r` : NDArray[np.complex128]
            The right unitary matrix.
        `U2l` : NDArray[np.complex128]
            The left unitary matrix.
        `U1r` : NDArray[np.complex128]
            The right unitary matrix.
        `U1l` : NDArray[np.complex128]
            The left unitary matrix.
        `U0r` : NDArray[np.complex128]
            The right unitary matrix.
        `U0l` : NDArray[np.complex128]
            The left unitary matrix.
        """
        # Calculate the decomposition
        U0l = weyl_decomposition.K1l.dot(U0L)
        U0r = weyl_decomposition.K1r.dot(U0R)
        U1l = U1L
        U1r = U1RA.dot(RZ(-2 * float(weyl_decomposition.c)).matrix).dot(UR1B)
        U2l = u2la.dot(RZ(-2 * float(weyl_decomposition.a)).matrix).dot(U2LB)
        U2r = U2RA.dot(RZ(2 * float(weyl_decomposition.b)).matrix).dot(U2RB)
        U3l = U3L.dot(weyl_decomposition.K2l)
        U3r = U3R.dot(weyl_decomposition.K2r)

        return U3r, U3l, U2r, U2l, U1r, U1l, U0r, U0l

    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:
        """ Apply the quantum unitary operator to a quantum circuit.

        Parameters
        ----------
        `circuit` : quick.circuit.Circuit
            The quantum circuit.
        `unitary` : NDArray[np.complex128] | quick.primitives.Operator
            The quantum unitary operator.
        `qubit_indices` : Sequence[int]
            The qubit indices to apply the unitary operator to. Note that
            the only reason the type hint is `int | Sequence[int]` is to
            not violate the parent class's method signature.

        Returns
        -------
        `circuit` : quick.circuit.Circuit
            The quantum circuit with the unitary operator applied.

        Raises
        ------
        ValueError
            - If the number of qubit indices is not equal to 2.
            - If the unitary matrix is not a 4x4 matrix.

        Usage
        -----
        >>> circuit = two_qubit_decomposition.apply_unitary(circuit, unitary, qubit_indices)
        """
        # Cast the qubit indices to a list if it is an integer
        # Note this is only to inform pylance that `qubit_indices` is a list
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        if len(qubit_indices) != 2:
            raise ValueError("Two-qubit decomposition requires exactly two qubit indices.")

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if unitary.num_qubits != 2:
            raise ValueError("Two-qubit decomposition requires a 4x4 unitary matrix.")

        decomposition_functions = [
            self._decomp0,
            self._decomp1,
            self._decomp2_supercontrolled,
            self._decomp3_supercontrolled,
        ]

        # Hardcoded global phase for supercontrolled basis ~Ud(pi/4, b, 0),
        # all b when using CX as KAK basis
        cx_basis_global_phase = -np.pi/4

        target_decomposed = TwoQubitWeylDecomposition(unitary.data)

        # Calculate the expected fidelities for different number of basis gates
        traces = self.traces(target_decomposed)
        expected_fidelities = [TwoQubitDecomposition.trace_to_fidelity(traces[i]) for i in range(4)]

        # Find the best number of basis gates
        best_num_basis = int(np.argmax(expected_fidelities))

        # Find the best decomposition
        decomposition = decomposition_functions[best_num_basis](target_decomposed)

        overall_global_phase = target_decomposed.global_phase - best_num_basis * cx_basis_global_phase

        if best_num_basis == 2:
            overall_global_phase += np.pi

        for i in range(best_num_basis):
            self.one_qubit_decomposition.apply_unitary(circuit, decomposition[2 * i], qubit_indices[0])
            self.one_qubit_decomposition.apply_unitary(circuit, decomposition[2 * i + 1], qubit_indices[1])
            circuit.CX(qubit_indices[0], qubit_indices[1])

        self.one_qubit_decomposition.apply_unitary(circuit, decomposition[2 * best_num_basis], qubit_indices[0])
        self.one_qubit_decomposition.apply_unitary(circuit, decomposition[2 * best_num_basis + 1], qubit_indices[1])

        circuit.GlobalPhase(overall_global_phase)

        return circuit

    def apply_unitary_up_to_diagonal(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> tuple[Circuit, NDArray[np.complex128]]:
        """ Apply the quantum unitary operator up to diagonal to a quantum circuit.

        Parameters
        ----------
        `circuit` : quick.circuit.Circuit
            The quantum circuit.
        `unitary` : NDArray[np.complex128] | quick.primitives.Operator
            The quantum unitary operator.
        `qubit_indices` : Sequence[int]
            The qubit indices to apply the unitary operator to. Note that
            the only reason the type hint is `int | Sequence[int]` is to
            not violate the parent class's method signature.

        Returns
        -------
        `circuit` : quick.circuit.Circuit
            The quantum circuit with the unitary operator applied.
        `diagonal` : NDArray[np.complex128]
            The diagonal matrix.

        Raises
        ------
        ValueError
            - If the number of qubit indices is not equal to 2.
            - If the unitary matrix is not a 4x4 matrix.

        Usage
        -----
        >>> circuit, diagonal = two_qubit_decomposition.apply_unitary_up_to_diagonal(circuit, unitary, qubit_indices)
        """
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        if len(qubit_indices) != 2:
            raise ValueError("Two-qubit decomposition requires exactly two qubit indices.")

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if unitary.num_qubits != 2:
            raise ValueError("Two-qubit decomposition requires a 4x4 unitary matrix.")

        su4, phase = TwoQubitDecomposition.u4_to_su4(unitary.data)
        diagonal = TwoQubitDecomposition.real_trace_transform(su4)
        mapped_su4 = diagonal @ su4

        circuit = self.apply_unitary(circuit, mapped_su4, qubit_indices)
        circuit.GlobalPhase(phase)

        return circuit, diagonal.conj()