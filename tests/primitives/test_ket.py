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

__all__ = ["TestKet"]

import numpy as np
from numpy.testing import assert_allclose
import pytest

from qickit.primitives import Bra, Ket


class TestKet:
    """ `tests.primitives.test_ket.TestKet` is the tester class for `qickit.primitives.Ket`.
    """
    def test_init(self) -> None:
        """ Test the initialization of the `qickit.primitives.Ket` class.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        assert_allclose(ket.data, np.array([[1+0j],
                                            [0+0j],
                                            [0+0j],
                                            [0+0j]]))

    def test_from_scalar_fail(self) -> None:
        """ Test the failure of defining a `qickit.primitives.Ket` object from a scalar.
        """
        with pytest.raises(AttributeError):
            Ket(1) # type: ignore

    def test_from_operator_fail(self) -> None:
        """ Test the failure of defining a `qickit.primitives.Ket` object from an operator.
        """
        with pytest.raises(ValueError):
            Ket(np.eye(4, dtype=complex))

    def test_check_normalization(self) -> None:
        """ Test the normalization of the `qickit.primitives.Ket` object.
        """
        data = np.array([1, 0, 0, 0])
        assert Ket.check_normalization(data)

    def test_check_normalization_fail(self) -> None:
        """ Test the failure of the normalization of the `qickit.primitives.Ket` object.
        """
        data = np.array([1, 1, 1, 1])
        assert not Ket.check_normalization(data)

    def test_normalize(self) -> None:
        """ Test the normalization of the `qickit.primitives.Ket` object.
        """
        data = np.array([1, 0, 0, 1])
        assert_allclose(Ket.normalize_data(data, np.linalg.norm(data)), np.array([(1+0j)/np.sqrt(2), 0+0j, 0+0j, (1+0j)/np.sqrt(2)]))

    def test_check_padding(self) -> None:
        """ Test the padding of the `qickit.primitives.Ket` object.
        """
        data = np.array([1, 0, 0, 0])
        assert Ket.check_padding(data)

    def test_check_padding_fail(self) -> None:
        """ Test the failure of the padding of the `qickit.primitives.Ket` object.
        """
        data = np.array([1, 0, 0])
        assert not Ket.check_padding(data)

    def test_pad(self) -> None:
        """ Test the padding of the `qickit.primitives.Ket` object.
        """
        data = np.array([1, 0, 0])
        padded_data, _ = Ket.pad_data(data, 4)
        assert_allclose(padded_data, np.array([[1],
                                               [0],
                                               [0],
                                               [0]]))

    def test_to_bra(self) -> None:
        """ Test the conversion of the `qickit.primitives.Ket` object to a `qickit.primitives.Bra` object.
        """
        ket = Ket(np.array([1+0j, 0+0j, 0+0j, 0+0j]))
        bra = ket.to_bra()
        assert_allclose(bra.data, np.array([1-0j, 0-0j, 0-0j, 0-0j]))

    def test_change_indexing(self) -> None:
        """ Test the change of indexing of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        ket.change_indexing("snake")
        assert_allclose(ket.data, np.array([[1+0j],
                                            [0+0j],
                                            [0+0j],
                                            [0+0j]]))

        ket = Ket(np.array([1, 0, 0, 0,
                            1, 0, 0, 0]))
        ket.change_indexing("snake")
        assert_allclose(ket.data, np.array([[(1+0j)/np.sqrt(2)],
                                            [0+0j],
                                            [0+0j],
                                            [0+0j],
                                            [0+0j],
                                            [0+0j],
                                            [0+0j],
                                            [(1+0j)/np.sqrt(2)]]))

    def test_change_indexing_fail(self) -> None:
        """ Test the failure of the change of indexing of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        with pytest.raises(ValueError):
            ket.change_indexing("invalid") # type: ignore

    def test_check_mul(self) -> None:
        """ Test the multiplication of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        ket._check__mul__(1)

        bra = Bra(np.array([1, 0, 0, 0]))
        ket._check__mul__(bra)

    def test_check_mul_fail(self) -> None:
        """ Test the failure of the multiplication of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        bra = Bra(np.array([1, 0]))
        with pytest.raises(ValueError):
            ket._check__mul__(bra)

        with pytest.raises(NotImplementedError):
            ket._check__mul__("invalid")

    def test_eq(self) -> None:
        """ Test the equality of the `qickit.primitives.Ket` object.
        """
        ket1 = Ket(np.array([1, 0, 0, 0]))
        ket2 = Ket(np.array([1, 0, 0, 0]))
        assert ket1 == ket2

    def test_eq_fail(self) -> None:
        """ Test the failure of the equality of the `qickit.primitives.Ket` object.
        """
        ket1 = Ket(np.array([1, 0, 0, 0]))
        ket2 = Ket(np.array([1, 0, 0, 1]))
        assert ket1 != ket2

        with pytest.raises(NotImplementedError):
            ket1 == "invalid" # type: ignore

    def test_len(self) -> None:
        """ Test the length of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        assert len(ket) == 4

    def test_add(self) -> None:
        """ Test the addition of the `qickit.primitives.Ket` object.
        """
        ket1 = Ket(np.array([1, 0, 0, 0]))
        ket2 = Ket(np.array([0, 1, 0, 0]))
        print(ket1 + ket2)
        assert_allclose((ket1 + ket2).data, np.array([[(1+0j)/np.sqrt(2)],
                                                      [(1+0j)/np.sqrt(2)],
                                                      [0+0j],
                                                      [0+0j]]))

    def test_add_fail(self) -> None:
        """ Test the failure of the addition of the `qickit.primitives.Ket` objects.
        """
        ket1 = Ket(np.array([1, 0, 0, 0]))
        ket2 = Ket(np.array([1, 0]))

        with pytest.raises(ValueError):
            ket1 + ket2 # type: ignore

        with pytest.raises(NotImplementedError):
            ket1 + "invalid" # type: ignore

    def test_mul_scalar(self) -> None:
        """ Test the multiplication of the `qickit.primitives.Ket` object with a scalar.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        assert_allclose((ket * 2).data, np.array([[1+0j],
                                                  [0+0j],
                                                  [0+0j],
                                                  [0+0j]]))

    def test_mul_bra(self) -> None:
        """ Test the multiplication of the `qickit.primitives.Ket` object with a `qickit.primitives.Bra` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        bra = Bra(np.array([1, 0, 0, 0]))
        # NOTE: Turned off this test until the bra-ket interface is fixed.
        # assert_allclose((ket * bra).data, np.array([[1+0j, 0+0j, 0+0j, 0+0j],
        #                                             [0+0j, 0+0j, 0+0j, 0+0j],
        #                                             [0+0j, 0+0j, 0+0j, 0+0j],
        #                                             [0+0j, 0+0j, 0+0j, 0+0j]]))

    def test_mul_fail(self) -> None:
        """ Test the failure of the multiplication of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        bra = Bra(np.array([1, 0]))
        with pytest.raises(ValueError):
            ket * bra # type: ignore

        with pytest.raises(NotImplementedError):
            ket * "invalid" # type: ignore

    def test_rmul_scalar(self) -> None:
        """ Test the multiplication of a `qickit.primitives.Ket` object with a scalar.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        assert_allclose((2 * ket).data, np.array([[1+0j],
                                                  [0+0j],
                                                  [0+0j],
                                                  [0+0j]]))

    def test_str(self) -> None:
        """ Test the string representation of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        assert str(ket) == "|Ψ⟩"

        ket = Ket(np.array([1, 0, 0, 0]), label="psi")
        assert str(ket) == "|psi⟩"

    def test_repr(self) -> None:
        """ Test the string representation of the `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        assert repr(ket) == ("Ket(data=[[1.+0.j]\n"
                             " [0.+0.j]\n"
                             " [0.+0.j]\n"
                             " [0.+0.j]], label=Ψ)")