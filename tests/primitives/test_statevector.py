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

from __future__ import annotations

__all__ = ["TestStatevector"]

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from quick.primitives import Statevector


class TestStatevector:
    """ `tests.primitives.test_statevector.TestStatevector` is the tester class
    for `quick.primitives.Statevector`.
    """
    def test_init(self) -> None:
        """ Test the initialization of the `quick.primitives.Statevector` class.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        assert_almost_equal(statevector.data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_from_scalar_fail(self) -> None:
        """ Test the failure of defining a `quick.primitives.Statevector` object from a scalar.
        """
        with pytest.raises(ValueError):
            Statevector(1) # type: ignore

    def test_from_operator_fail(self) -> None:
        """ Test the failure of defining a `quick.primitives.Statevector` object from an operator.
        """
        with pytest.raises(ValueError):
            Statevector(np.eye(4, dtype=complex))

    def test_check_normalization(self) -> None:
        """ Test the normalization of the `quick.primitives.Statevector` object.
        """
        data = np.array([1, 0, 0, 0])
        assert Statevector.check_normalization(data)

    def test_check_normalization_fail(self) -> None:
        """ Test the failure of the normalization of the `quick.primitives.Statevector` object.
        """
        data = np.array([1, 1, 1, 1])
        assert not Statevector.check_normalization(data)

    def test_normalize(self) -> None:
        """ Test the normalization of the `quick.primitives.Statevector` object.
        """
        data = np.array([1, 0, 0, 1])
        assert_almost_equal(
            Statevector.normalize_data(
                data,
                np.linalg.norm(data)), np.array([(1+0j)/np.sqrt(2), 0+0j, 0+0j, (1+0j)/np.sqrt(2)]
            )
        )

        statevector = Statevector(data)
        statevector.normalize()
        assert_almost_equal(statevector.data, np.array([(1+0j)/np.sqrt(2), 0+0j, 0+0j, (1+0j)/np.sqrt(2)]))

        # Re-normalize the already normalized to cover the case where if normalized we simply return
        statevector.normalize()
        assert_almost_equal(statevector.data, np.array([(1+0j)/np.sqrt(2), 0+0j, 0+0j, (1+0j)/np.sqrt(2)]))

    def test_check_padding(self) -> None:
        """ Test the padding of the `quick.primitives.Statevector` object.
        """
        data = np.array([1, 0, 0, 0])
        assert Statevector.check_padding(data)

    def test_check_padding_fail(self) -> None:
        """ Test the failure of the padding of the `quick.primitives.Statevector` object.
        """
        data = np.array([1, 0, 0])
        assert not Statevector.check_padding(data)

    def test_pad(self) -> None:
        """ Test the padding of the `quick.primitives.Statevector` object.
        """
        data = np.array([1, 0, 0])
        padded_data = Statevector.pad_data(data, 4)
        assert_almost_equal(padded_data, np.array([1, 0, 0, 0]))

        statevector = Statevector(data)
        statevector.pad()
        assert_almost_equal(statevector.data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

        # Re-pad the already padded to cover the case where if padded we simply return
        statevector.pad()
        assert_almost_equal(statevector.data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_change_indexing(self) -> None:
        """ Test the change of indexing of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        statevector.change_indexing("snake")
        assert_almost_equal(statevector.data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

        statevector = Statevector(np.array([1, 0, 0, 0,
                            1, 0, 0, 0]))
        statevector.change_indexing("snake")
        assert_almost_equal(statevector.data, np.array([
            (1+0j)/np.sqrt(2), 0+0j, 0+0j, 0+0j,
            0+0j, 0+0j, 0+0j, (1+0j)/np.sqrt(2)
        ]))

    def test_change_indexing_fail(self) -> None:
        """ Test the failure of the change of indexing of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        with pytest.raises(ValueError):
            statevector.change_indexing("invalid") # type: ignore

    def test_check_mul(self) -> None:
        """ Test the multiplication of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        statevector._check__mul__(1)

    def test_check_mul_fail(self) -> None:
        """ Test the failure of the multiplication of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))

        with pytest.raises(TypeError):
            statevector._check__mul__("invalid")

    def test_eq(self) -> None:
        """ Test the equality of the `quick.primitives.Statevector` object.
        """
        statevector1 = Statevector(np.array([1, 0, 0, 0]))
        statevector2 = Statevector(np.array([1, 0, 0, 0]))
        assert statevector1 == statevector2

    def test_eq_fail(self) -> None:
        """ Test the failure of the equality of the `quick.primitives.Statevector` object.
        """
        statevector1 = Statevector(np.array([1, 0, 0, 0]))
        statevector2 = Statevector(np.array([0, 1, 0, 0]))
        assert statevector1 != statevector2

        with pytest.raises(TypeError):
            statevector1 == "invalid" # type: ignore

    def test_len(self) -> None:
        """ Test the length of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        assert len(statevector) == 4

    def test_add(self) -> None:
        """ Test the addition of the `quick.primitives.Statevector` objects.
        """
        statevector1 = Statevector(np.array([1, 0, 0, 0]))
        statevector2 = Statevector(np.array([0, 1, 0, 0]))
        assert_almost_equal(
            (statevector1 + statevector2).data,
            np.array([(1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2), 0+0j, 0+0j])
        )

    def test_add_fail(self) -> None:
        """ Test the failure of the addition of the `quick.primitives.Statevector` objects.
        """
        statevector1 = Statevector(np.array([1, 0, 0, 0]))
        statevector2 = Statevector(np.array([1, 0]))

        with pytest.raises(ValueError):
            statevector1 + statevector2 # type: ignore

        with pytest.raises(TypeError):
            statevector1 + "invalid" # type: ignore

    def test_mul_scalar(self) -> None:
        """ Test the multiplication of the `quick.primitives.Statevector` object with a scalar.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        assert_almost_equal((statevector * 2).data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_mul_fail(self) -> None:
        """ Test the failure of the multiplication of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0]))

        with pytest.raises(TypeError):
            statevector * "invalid" # type: ignore

    def test_rmul_scalar(self) -> None:
        """ Test the multiplication of a `quick.primitives.Statevector` object with a scalar.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        assert_almost_equal((2 * statevector).data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_str(self) -> None:
        """ Test the string representation of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        assert str(statevector) == "|Ψ⟩"

        statevector = Statevector(np.array([1, 0, 0, 0]), label="psi")
        assert str(statevector) == "|psi⟩"

    def test_repr(self) -> None:
        """ Test the string representation of the `quick.primitives.Statevector` object.
        """
        statevector = Statevector(np.array([1, 0, 0, 0]))
        assert repr(statevector) == "Statevector(data=[1.+0.j 0.+0.j 0.+0.j 0.+0.j], label=Ψ)"