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

__all__ = ["TestBra"]

import numpy as np
from numpy.testing import assert_allclose
import pytest

from qickit.primitives import Bra, Ket, Operator


class TestBra:
    """ `tests.primitives.test_bra.TestBra` is the tester class for `qickit.primitives.Bra`.
    """
    def test_init(self) -> None:
        """ Test the initialization of the `qickit.primitives.Bra` class.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        assert_allclose(bra.data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_from_scalar_fail(self) -> None:
        """ Test the failure of defining a `qickit.primitives.Bra` object from a scalar.
        """
        with pytest.raises(AttributeError):
            Bra(1) # type: ignore

    def test_from_operator_fail(self) -> None:
        """ Test the failure of defining a `qickit.primitives.Bra` object from an operator.
        """
        with pytest.raises(ValueError):
            Bra(np.eye(4, dtype=complex))

    def test_check_normalization(self) -> None:
        """ Test the normalization of the `qickit.primitives.Bra` object.
        """
        data = np.array([1, 0, 0, 0])
        assert Bra.check_normalization(data)

    def test_check_normalization_fail(self) -> None:
        """ Test the failure of the normalization of the `qickit.primitives.Bra` object.
        """
        data = np.array([1, 1, 1, 1])
        assert not Bra.check_normalization(data)

    def test_normalize(self) -> None:
        """ Test the normalization of the `qickit.primitives.Bra` object.
        """
        data = np.array([1, 0, 0, 1])
        assert_allclose(Bra.normalize_data(data, np.linalg.norm(data)), np.array([(1+0j)/np.sqrt(2), 0+0j, 0+0j, (1+0j)/np.sqrt(2)]))

    def test_check_padding(self) -> None:
        """ Test the padding of the `qickit.primitives.Bra` object.
        """
        data = np.array([1, 0, 0, 0])
        assert Bra.check_padding(data)

    def test_check_padding_fail(self) -> None:
        """ Test the failure of the padding of the `qickit.primitives.Bra` object.
        """
        data = np.array([1, 0, 0])
        assert not Bra.check_padding(data)

    def test_pad(self) -> None:
        """ Test the padding of the `qickit.primitives.Bra` object.
        """
        data = np.array([1, 0, 0])
        padded_data, _ = Bra.pad_data(data, 4)
        assert_allclose(padded_data, np.array([1, 0, 0, 0]))

    def test_to_ket(self) -> None:
        """ Test the conversion of the `qickit.primitives.Bra` object to a `qickit.primitives.Ket` object.
        """
        bra = Bra(np.array([1+0j, 0+0j, 0+0j, 0+0j]))
        ket = bra.to_ket()
        assert_allclose(ket.data, np.array([[1-0j],
                                            [0-0j],
                                            [0-0j],
                                            [0-0j]]))

    def test_change_indexing(self) -> None:
        """ Test the change of indexing of the `qickit.primitives.Bra` object.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        bra.change_indexing("snake")
        assert_allclose(bra.data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

        bra = Bra(np.array([1, 0, 0, 0,
                            1, 0, 0, 0]))
        bra.change_indexing("snake")
        assert_allclose(bra.data, np.array([(1+0j)/np.sqrt(2), 0+0j, 0+0j, 0+0j,
                                            0+0j, 0+0j, 0+0j, (1+0j)/np.sqrt(2)]))

    def test_change_indexing_fail(self) -> None:
        """ Test the failure of the change of indexing of the `qickit.primitives.Bra` object.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        with pytest.raises(ValueError):
            bra.change_indexing("invalid") # type: ignore

    def test_check_mul(self) -> None:
        """ Test the multiplication of the `qickit.primitives.Bra` object.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        bra._check__mul__(1)

        ket = Ket(np.array([1, 0, 0, 0]))
        bra._check__mul__(ket)

        operator = Operator(np.eye(4, dtype=complex))
        bra._check__mul__(operator)

    def test_check_mul_fail(self) -> None:
        """ Test the failure of the multiplication of the `qickit.primitives.Bra` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        bra = Bra(np.array([1, 0]))
        with pytest.raises(ValueError):
            bra._check__mul__(ket)

        operator = Operator(np.eye(4, dtype=complex))
        with pytest.raises(ValueError):
            bra._check__mul__(operator)

        with pytest.raises(NotImplementedError):
            bra._check__mul__("invalid")

    def test_eq(self) -> None:
        """ Test the equality of the `qickit.primitives.Bra` object.
        """
        bra1 = Bra(np.array([1, 0, 0, 0]))
        bra2 = Bra(np.array([1, 0, 0, 0]))
        assert bra1 == bra2

    def test_eq_fail(self) -> None:
        """ Test the failure of the equality of the `qickit.primitives.Bra` object.
        """
        bra1 = Bra(np.array([1, 0, 0, 0]))
        bra2 = Bra(np.array([0, 1, 0, 0]))
        assert bra1 != bra2

        with pytest.raises(NotImplementedError):
            bra1 == "invalid" # type: ignore

    def test_len(self) -> None:
        """ Test the length of the `qickit.primitives.Bra` object.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        assert len(bra) == 4

    def test_add(self) -> None:
        """ Test the addition of the `qickit.primitives.Bra` objects.
        """
        bra1 = Bra(np.array([1, 0, 0, 0]))
        bra2 = Bra(np.array([0, 1, 0, 0]))
        assert_allclose((bra1 + bra2).data, np.array([(1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2), 0+0j, 0+0j]))

    def test_add_fail(self) -> None:
        """ Test the failure of the addition of the `qickit.primitives.Bra` objects.
        """
        bra1 = Bra(np.array([1, 0, 0, 0]))
        bra2 = Bra(np.array([1, 0]))

        with pytest.raises(ValueError):
            bra1 + bra2 # type: ignore

        with pytest.raises(NotImplementedError):
            bra1 + "invalid" # type: ignore

    def test_mul_scalar(self) -> None:
        """ Test the multiplication of the `qickit.primitives.Bra` object with a scalar.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        assert_allclose((bra * 2).data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_mul_bra(self) -> None:
        """ Test the multiplication of the `qickit.primitives.Bra` object with a `qickit.primitives.Ket` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        bra = Bra(np.array([1, 0, 0, 0]))
        assert bra * ket == 1.0 + 0j

    def test_mul_fail(self) -> None:
        """ Test the failure of the multiplication of the `qickit.primitives.Bra` object.
        """
        ket = Ket(np.array([1, 0, 0, 0]))
        bra = Bra(np.array([1, 0]))
        with pytest.raises(ValueError):
            bra * ket # type: ignore

        with pytest.raises(NotImplementedError):
            bra * "invalid" # type: ignore

    def test_rmul_scalar(self) -> None:
        """ Test the multiplication of a `qickit.primitives.Bra` object with a scalar.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        assert_allclose((2 * bra).data, np.array([1+0j, 0+0j, 0+0j, 0+0j]))

    def test_str(self) -> None:
        """ Test the string representation of the `qickit.primitives.Bra` object.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        assert str(bra) == "⟨Ψ|"

        bra = Bra(np.array([1, 0, 0, 0]), label="psi")
        assert str(bra) == "⟨psi|"

    def test_repr(self) -> None:
        """ Test the string representation of the `qickit.primitives.Bra` object.
        """
        bra = Bra(np.array([1, 0, 0, 0]))
        print(repr(bra))
        assert repr(bra) == "Bra(data=[1.+0.j 0.+0.j 0.+0.j 0.+0.j], label=Ψ)"