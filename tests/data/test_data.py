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

__all__ = ['test_data']

from collections.abc import MutableSequence
import numpy as np

# QICKIT imports
from qickit.data import Data


def test_data() -> None:
    """ Test the `qickit.data.Data` class.
    """
    data = Data([[1, 2, 3], [4, 5, 6]])
    assert data.normalized is False
    assert data.padded is False
    assert data.shape == (2, 3)

    def is_normalized(state: MutableSequence[MutableSequence[float]] | MutableSequence[float]) -> bool:
        """ Test if a state vector is normalized.

        Parameters
        -----------
        state (MutableSequence):
            A list representing a state vector.

        Returns
        --------
        (bool): True if the state vector is normalized. False otherwise.
        """
        # Calculate the norm squared of the state vector
        norm_squared = np.linalg.norm(state) ** 2

        # Set the tolerance
        epsilon = 1e-6

        # Assert the Born rule
        if np.abs(norm_squared - 1) < epsilon:
            return True
        else:
            return False

    data.normalize()
    assert data.normalized
    assert is_normalized(data.data)
    assert data.shape == (2, 3)

    data.pad()

    assert data.padded
    assert data.shape == (2, 4)
    assert data.num_qubits == 3