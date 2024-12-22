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

""" One-qubit unitary decomposition using ZYZ and U3 basis.
"""

from __future__ import annotations

__all__ = ["OneQubitDecomposition"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, SupportsIndex, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Operator
from qickit.synthesis.unitarypreparation import UnitaryPreparation


class OneQubitDecomposition(UnitaryPreparation):
    """ `qickit.synthesis.unitarypreparation.OneQubitDecomposition` is the class for decomposing
    one-qubit unitary matrices into one qubit quantum gates.

    Notes
    -----
    The one-qubit decomposition is based on the ZYZ or U3 decomposition of a 2x2 unitary matrix.
    The class is a special case of the `UnitaryPreparation` class, where the unitary matrix must
    be a 2x2 matrix.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `basis` : Literal["zyz", "u3"], optional, default="u3"
        The basis for the decomposition.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `basis` : Literal["zyz", "u3"]
        The basis for the decomposition.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `qickit.circuit.Circuit`.
    ValueError
        - If the basis is not supported.

    Usage
    -----
    >>> one_qubit_decomposition = OneQubitDecomposition(output_framework=Circuit, basis="zyz")
    """
    def __init__(
            self,
            output_framework: type[Circuit],
            basis: Literal["zyz", "u3"]="u3"
        ) -> None:

        super().__init__(output_framework)

        if basis not in ["zyz", "u3"]:
            raise ValueError(f"{basis} is not a supported decomposition method of ['zyz', 'u3'].")
        self.basis = basis

    @staticmethod
    def one_qubit_det(U: NDArray[np.complex128]) -> float:
        """ Calculate the determinant of a 2x2 unitary matrix.

        Parameters
        ----------
        `U` : NDArray[np.complex128]
            2x2 unitary matrix.

        Returns
        -------
        `det` : float
            The determinant of the unitary matrix.
        """
        return U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]

    @staticmethod
    def params_zyz(U: NDArray[np.complex128]) -> tuple[float, tuple[float, float, float]]:
        """ Calculate the ZYZ decomposition of a 2x2 unitary matrix.

        Notes
        -----
        The ZYZ decomposition of a 2x2 unitary matrix :math:`U` is given by

        .. math::
            U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)

        Parameters
        ----------
        `U` : NDArray[np.complex128]
            2x2 unitary matrix.

        Returns
        -------
        `alpha` : float
            The global phase.
        `theta` : float
            The angle of the Y rotation.
        `phi` : float
            The angle of the first Z rotation.
        `lam` : float
            The angle of the second Z rotation.
        """
        det_arg = OneQubitDecomposition.one_qubit_det(U)

        # Calculate the principal argument of the determinant
        det_arg = np.arctan2(det_arg.imag, det_arg.real)

        # Calculate the global phase
        alpha = 0.5 * det_arg

        # Calculate the ZYZ decomposition
        theta = 2. * np.arctan2(np.abs(U[1, 0]), np.abs(U[0, 0]))
        angle_1 = np.arctan2(U[1, 1].imag, U[1, 1].real)
        angle_2 = np.arctan2(U[1, 0].imag, U[1, 0].real)
        phi = angle_1 + angle_2 - det_arg
        lam = angle_1 - angle_2

        return alpha, (theta, phi, lam)

    @staticmethod
    def params_u3(U: NDArray[np.complex128]) -> tuple[float, tuple[float, float, float]]:
        """ Calculate the U3 parameters to implement a 2x2 unitary matrix.

        Notes
        -----
        Given a 2x2 unitary matrix :math:`U`, this function returns the three parameters

        .. math::
            U = exp(i p) U3(\theta, \phi, \lambda)

        Parameters
        ----------
        `U` : NDArray[np.complex128]
            2x2 unitary matrix.

        Returns
        -------
        `theta` : float
            The angle of the Y rotation.
        `phi` : float
            The angle of the first Z rotation.
        `lam` : float
            The angle of the second Z rotation.
        `phase` : float
            The global phase.
        """
        alpha, (theta, phi, lam) = OneQubitDecomposition.params_zyz(U)
        phase = alpha - (phi + lam) / 2
        return phase, (theta, phi, lam)

    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:
        """ Apply the quantum unitary operator to a quantum circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The quantum unitary operator.
        `qubit_indices` : int | Sequence[int]
            The qubit indices to apply the unitary operator to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit with the unitary operator applied.

        Raises
        ------
        ValueError
            - If the number of qubit indices is not equal to 1.
            - If the unitary matrix is not a 2x2 matrix.
        """
        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        if len(qubit_indices) != 1:
            raise ValueError("One-qubit decomposition requires exactly one qubit index.")

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if unitary.num_qubits != 1:
            raise ValueError("One-qubit decomposition requires a 2x2 unitary matrix.")

        if self.basis == "zyz":
            alpha, (theta, phi, lamda) = OneQubitDecomposition.params_zyz(unitary.data)
            circuit.RZ(lamda, qubit_indices[0])
            circuit.RY(theta, qubit_indices[0])
            circuit.RZ(phi, qubit_indices[0])
            circuit.GlobalPhase(alpha)

        elif self.basis == "u3":
            phase, (theta, phi, lamda) = OneQubitDecomposition.params_u3(unitary.data)
            circuit.U3([theta, phi, lamda], qubit_indices[0])
            circuit.GlobalPhase(phase)

        return circuit