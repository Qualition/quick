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

""" Utility functions for two-qubit gate decompositions.
"""

from __future__ import annotations

__all__ = ["u4_to_su4"]

import cmath
import numpy as np
from numpy.typing import NDArray
import scipy.linalg # type: ignore


def u4_to_su4(U: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], float]:
    """ Convert a U(4) matrix to an SU(4) matrix by removing the global phase
    such that the determinant is 1.

    Parameters
    ----------
    `U` : NDArray[np.complex128]
        The input U(4) matrix.

    Returns
    -------
    `SU4` : NDArray[np.complex128]
        The resulting SU(4) matrix.
    `global_phase` : float
        The global phase that was removed.

    Usage
    -----
    >>> SU4, global_phase = u4_to_su4(np.eye(4))
    """
    # We need to cast to complex to avoid NaN errors
    U = np.asarray(U, dtype=np.complex128)

    # The code fails with np.linalg.det
    U_det = scipy.linalg.det(U)

    # For general U_N we must take to power of -1/U.shape[0]
    # but since this implementation is only used for U4 we
    # omit this calculation and use hardcoded -1/4
    SU4 = U * U_det ** (-0.25)
    global_phase = cmath.phase(U_det) / 4

    return SU4, global_phase