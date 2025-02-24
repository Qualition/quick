# Copyright 2023-2025 Qualition Computing LLC.
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

import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from quick.circuit import Circuit
from quick.primitives import Operator
from quick.synthesis.gate_decompositions import OneQubitDecomposition
from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import TwoQubitWeylDecomposition
from quick.synthesis.unitarypreparation import UnitaryPreparation

__all__ = ["TwoQubitDecomposition"]

class TwoQubitDecomposition(UnitaryPreparation):
    one_qubit_decomposition: OneQubitDecomposition
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    @staticmethod
    def u4_to_su4(u4: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], float]: ...
    @staticmethod
    def traces(target: TwoQubitWeylDecomposition) -> list[complex]: ...
    @staticmethod
    def real_trace_transform(U: NDArray[np.complex128]) -> NDArray[np.complex128]: ...
    @staticmethod
    def trace_to_fidelity(trace: complex) -> float: ...
    @staticmethod
    def _decomp0(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]: ...
    @staticmethod
    def _decomp1(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128]
        ]: ...
    @staticmethod
    def _decomp2_supercontrolled(weyl_decomposition: TwoQubitWeylDecomposition) -> tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128]
        ]: ...
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
        ]: ...
    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit: ...
    def apply_unitary_up_to_diagonal(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> tuple[Circuit, NDArray[np.complex128]]: ...
