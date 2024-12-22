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

__all__ = ["TestTKETCircuit"]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
import pytest

from pytket import Circuit as TKCircuit

from qickit.circuit import TKETCircuit
from qickit.circuit.gate_matrix import RX, RY, RZ, Phase, U3

from tests.circuit import Template
from tests.circuit.utils import cosine_similarity
from tests.circuit.gate_utils import (
    X_unitary_matrix,
    Y_unitary_matrix,
    Z_unitary_matrix,
    H_unitary_matrix,
    S_unitary_matrix,
    T_unitary_matrix,
    XPow_unitary_matrix,
    XPow_global_shift_unitary_matrix,
    YPow_unitary_matrix,
    YPow_global_shift_unitary_matrix,
    ZPow_unitary_matrix,
    ZPow_global_shift_unitary_matrix,
    RXX_unitary_matrix_pi_over_4_01qubits,
    RXX_unitary_matrix_pi_over_4_10qubits,
    RXX_unitary_matrix_1_over_4_02qubits,
    RXX_unitary_matrix_1_over_4_20qubits,
    RXX_unitary_matrix_pi_over_4_12qubits,
    RYY_unitary_matrix_pi_over_4_01qubits,
    RYY_unitary_matrix_pi_over_4_10qubits,
    RYY_unitary_matrix_1_over_4_02qubits,
    RYY_unitary_matrix_1_over_4_20qubits,
    RYY_unitary_matrix_pi_over_4_12qubits,
    RZZ_unitary_matrix_pi_over_4_01qubits,
    RZZ_unitary_matrix_pi_over_4_10qubits,
    RZZ_unitary_matrix_1_over_4_02qubits,
    RZZ_unitary_matrix_1_over_4_20qubits,
    RZZ_unitary_matrix_pi_over_4_12qubits,
    SWAP_unitary_matrix_01qubits,
    SWAP_unitary_matrix_10qubits,
    SWAP_unitary_matrix_02qubits,
    SWAP_unitary_matrix_20qubits,
    SWAP_unitary_matrix_12qubits,
    CX_unitary_matrix_01qubits,
    CX_unitary_matrix_10qubits,
    CX_unitary_matrix_02qubits,
    CX_unitary_matrix_20qubits,
    CX_unitary_matrix_12qubits,
    CY_unitary_matrix_01qubits,
    CY_unitary_matrix_10qubits,
    CY_unitary_matrix_02qubits,
    CY_unitary_matrix_20qubits,
    CY_unitary_matrix_12qubits,
    CZ_unitary_matrix_01qubits,
    CZ_unitary_matrix_10qubits,
    CZ_unitary_matrix_02qubits,
    CZ_unitary_matrix_20qubits,
    CZ_unitary_matrix_12qubits,
    CH_unitary_matrix_01qubits,
    CH_unitary_matrix_10qubits,
    CH_unitary_matrix_02qubits,
    CH_unitary_matrix_20qubits,
    CH_unitary_matrix_12qubits,
    CS_unitary_matrix_01qubits,
    CS_unitary_matrix_10qubits,
    CS_unitary_matrix_02qubits,
    CS_unitary_matrix_20qubits,
    CS_unitary_matrix_12qubits,
    CT_unitary_matrix_01qubits,
    CT_unitary_matrix_10qubits,
    CT_unitary_matrix_02qubits,
    CT_unitary_matrix_20qubits,
    CT_unitary_matrix_12qubits,
    CSdg_unitary_matrix_01qubits,
    CSdg_unitary_matrix_10qubits,
    CSdg_unitary_matrix_02qubits,
    CSdg_unitary_matrix_20qubits,
    CSdg_unitary_matrix_12qubits,
    CTdg_unitary_matrix_01qubits,
    CTdg_unitary_matrix_10qubits,
    CTdg_unitary_matrix_02qubits,
    CTdg_unitary_matrix_20qubits,
    CTdg_unitary_matrix_12qubits,
    CRX_unitary_matrix_pi_over_4_01qubits,
    CRX_unitary_matrix_pi_over_4_10qubits,
    CRX_unitary_matrix_1_over_4_02qubits,
    CRX_unitary_matrix_1_over_4_20qubits,
    CRX_unitary_matrix_pi_over_4_12qubits,
    CRY_unitary_matrix_pi_over_4_01qubits,
    CRY_unitary_matrix_pi_over_4_10qubits,
    CRY_unitary_matrix_1_over_4_02qubits,
    CRY_unitary_matrix_1_over_4_20qubits,
    CRY_unitary_matrix_pi_over_4_12qubits,
    CRZ_unitary_matrix_pi_over_4_01qubits,
    CRZ_unitary_matrix_pi_over_4_10qubits,
    CRZ_unitary_matrix_1_over_4_02qubits,
    CRZ_unitary_matrix_1_over_4_20qubits,
    CRZ_unitary_matrix_pi_over_4_12qubits,
    CPhase_unitary_matrix_pi_over_4_01qubits,
    CPhase_unitary_matrix_pi_over_4_10qubits,
    CPhase_unitary_matrix_1_over_4_02qubits,
    CPhase_unitary_matrix_1_over_4_20qubits,
    CPhase_unitary_matrix_pi_over_4_12qubits,
    CXPow_unitary_matrix_1_over_4_0_shift_01qubits,
    CXPow_unitary_matrix_1_over_4_0_shift_10qubits,
    CXPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits,
    CXPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits,
    CXPow_unitary_matrix_negative1_over_4_0_shift_12qubits,
    CYPow_unitary_matrix_1_over_4_0_shift_01qubits,
    CYPow_unitary_matrix_1_over_4_0_shift_10qubits,
    CYPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits,
    CYPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits,
    CYPow_unitary_matrix_negative1_over_4_0_shift_12qubits,
    CZPow_unitary_matrix_1_over_4_0_shift_01qubits,
    CZPow_unitary_matrix_1_over_4_0_shift_10qubits,
    CZPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits,
    CZPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits,
    CZPow_unitary_matrix_negative1_over_4_0_shift_12qubits,
    CRXX_unitary_matrix_pi_over_4_012qubits,
    CRXX_unitary_matrix_pi_over_4_102qubits,
    CRXX_unitary_matrix_1_over_4_123qubits,
    CRXX_unitary_matrix_1_over_4_213qubits,
    CRXX_unitary_matrix_pi_over_4_023qubits,
    CRYY_unitary_matrix_pi_over_4_012qubits,
    CRYY_unitary_matrix_pi_over_4_102qubits,
    CRYY_unitary_matrix_1_over_4_123qubits,
    CRYY_unitary_matrix_1_over_4_213qubits,
    CRYY_unitary_matrix_pi_over_4_023qubits,
    CRZZ_unitary_matrix_pi_over_4_012qubits,
    CRZZ_unitary_matrix_pi_over_4_102qubits,
    CRZZ_unitary_matrix_1_over_4_123qubits,
    CRZZ_unitary_matrix_1_over_4_213qubits,
    CRZZ_unitary_matrix_pi_over_4_023qubits,
    CU3_unitary_matrix_pi2_pi3_pi4_01qubits,
    CU3_unitary_matrix_pi2_pi3_pi4_10qubits,
    CU3_unitary_matrix_pi2_pi3_pi4_02qubits,
    CU3_unitary_matrix_pi2_pi3_pi4_20qubits,
    CU3_unitary_matrix_pi2_pi3_pi4_12qubits,
    CSWAP_unitary_matrix_012qubits,
    CSWAP_unitary_matrix_102qubits,
    CSWAP_unitary_matrix_123qubits,
    CSWAP_unitary_matrix_213qubits,
    CSWAP_unitary_matrix_023qubits,
    MCX_unitary_matrix_01_23_qubits,
    MCX_unitary_matrix_10_23_qubits,
    MCX_unitary_matrix_02_13_qubits,
    MCX_unitary_matrix_20_34_qubits,
    MCX_unitary_matrix_12_04_qubits,
    MCX_unitary_matrix_53_01_qubits,
    MCX_unitary_matrix_012_34_qubits,
    MCX_unitary_matrix_01_234_qubits,
    MCX_unitary_matrix_012_345_qubits,
    MCX_unitary_matrix_01_2_qubits,
    MCX_unitary_matrix_0_23_qubits,
    MCY_unitary_matrix_01_23_qubits,
    MCY_unitary_matrix_10_23_qubits,
    MCY_unitary_matrix_02_13_qubits,
    MCY_unitary_matrix_20_34_qubits,
    MCY_unitary_matrix_12_04_qubits,
    MCY_unitary_matrix_53_01_qubits,
    MCY_unitary_matrix_012_34_qubits,
    MCY_unitary_matrix_01_234_qubits,
    MCY_unitary_matrix_012_345_qubits,
    MCY_unitary_matrix_01_2_qubits,
    MCY_unitary_matrix_0_23_qubits,
    MCZ_unitary_matrix_01_23_qubits,
    MCZ_unitary_matrix_10_23_qubits,
    MCZ_unitary_matrix_02_13_qubits,
    MCZ_unitary_matrix_20_34_qubits,
    MCZ_unitary_matrix_12_04_qubits,
    MCZ_unitary_matrix_53_01_qubits,
    MCZ_unitary_matrix_012_34_qubits,
    MCZ_unitary_matrix_01_234_qubits,
    MCZ_unitary_matrix_012_345_qubits,
    MCZ_unitary_matrix_01_2_qubits,
    MCZ_unitary_matrix_0_23_qubits,
    MCH_unitary_matrix_01_23_qubits,
    MCH_unitary_matrix_10_23_qubits,
    MCH_unitary_matrix_02_13_qubits,
    MCH_unitary_matrix_20_34_qubits,
    MCH_unitary_matrix_12_04_qubits,
    MCH_unitary_matrix_53_01_qubits,
    MCH_unitary_matrix_012_34_qubits,
    MCH_unitary_matrix_01_234_qubits,
    MCH_unitary_matrix_012_345_qubits,
    MCH_unitary_matrix_01_2_qubits,
    MCH_unitary_matrix_0_23_qubits,
    MCS_unitary_matrix_01_23_qubits,
    MCS_unitary_matrix_10_23_qubits,
    MCS_unitary_matrix_02_13_qubits,
    MCS_unitary_matrix_20_34_qubits,
    MCS_unitary_matrix_12_04_qubits,
    MCS_unitary_matrix_53_01_qubits,
    MCS_unitary_matrix_012_34_qubits,
    MCS_unitary_matrix_01_234_qubits,
    MCS_unitary_matrix_012_345_qubits,
    MCS_unitary_matrix_01_2_qubits,
    MCS_unitary_matrix_0_23_qubits,
    MCT_unitary_matrix_01_23_qubits,
    MCT_unitary_matrix_10_23_qubits,
    MCT_unitary_matrix_02_13_qubits,
    MCT_unitary_matrix_20_34_qubits,
    MCT_unitary_matrix_12_04_qubits,
    MCT_unitary_matrix_53_01_qubits,
    MCT_unitary_matrix_012_34_qubits,
    MCT_unitary_matrix_01_234_qubits,
    MCT_unitary_matrix_012_345_qubits,
    MCT_unitary_matrix_01_2_qubits,
    MCT_unitary_matrix_0_23_qubits,
    MCSdg_unitary_matrix_01_23_qubits,
    MCSdg_unitary_matrix_10_23_qubits,
    MCSdg_unitary_matrix_02_13_qubits,
    MCSdg_unitary_matrix_20_34_qubits,
    MCSdg_unitary_matrix_12_04_qubits,
    MCSdg_unitary_matrix_53_01_qubits,
    MCSdg_unitary_matrix_012_34_qubits,
    MCSdg_unitary_matrix_01_234_qubits,
    MCSdg_unitary_matrix_012_345_qubits,
    MCSdg_unitary_matrix_01_2_qubits,
    MCSdg_unitary_matrix_0_23_qubits,
    MCTdg_unitary_matrix_01_23_qubits,
    MCTdg_unitary_matrix_10_23_qubits,
    MCTdg_unitary_matrix_02_13_qubits,
    MCTdg_unitary_matrix_20_34_qubits,
    MCTdg_unitary_matrix_12_04_qubits,
    MCTdg_unitary_matrix_53_01_qubits,
    MCTdg_unitary_matrix_012_34_qubits,
    MCTdg_unitary_matrix_01_234_qubits,
    MCTdg_unitary_matrix_012_345_qubits,
    MCTdg_unitary_matrix_01_2_qubits,
    MCTdg_unitary_matrix_0_23_qubits,
    MCRX_unitary_matrix_pi_over_4_01_23_qubits,
    MCRX_unitary_matrix_pi_over_4_10_23_qubits,
    MCRX_unitary_matrix_1_over_4_02_13_qubits,
    MCRX_unitary_matrix_1_over_4_20_34_qubits,
    MCRX_unitary_matrix_negative1_over_4_12_04_qubits,
    MCRX_unitary_matrix_negative1_over_4_53_01_qubits,
    MCRX_unitary_matrix_1_over_3_012_34_qubits,
    MCRX_unitary_matrix_1_over_3_01_234_qubits,
    MCRX_unitary_matrix_pi_over_4_012_345_qubits,
    MCRX_unitary_matrix_pi_over_4_01_2_qubits,
    MCRX_unitary_matrix_pi_over_4_0_23_qubits,
    MCRX_unitary_matrix_0dot1_012345_6_qubits,
    MCRX_unitary_matrix_0dot1_0123456_7_qubits,
    MCRX_unitary_matrix_0dot1_01234567_8_qubits,
    MCRY_unitary_matrix_pi_over_4_01_23_qubits,
    MCRY_unitary_matrix_pi_over_4_10_23_qubits,
    MCRY_unitary_matrix_1_over_4_02_13_qubits,
    MCRY_unitary_matrix_1_over_4_20_34_qubits,
    MCRY_unitary_matrix_negative1_over_4_12_04_qubits,
    MCRY_unitary_matrix_negative1_over_4_53_01_qubits,
    MCRY_unitary_matrix_1_over_3_012_34_qubits,
    MCRY_unitary_matrix_1_over_3_01_234_qubits,
    MCRY_unitary_matrix_pi_over_4_012_345_qubits,
    MCRY_unitary_matrix_pi_over_4_01_2_qubits,
    MCRY_unitary_matrix_pi_over_4_0_23_qubits,
    MCRY_unitary_matrix_0dot1_012345_6_qubits,
    MCRY_unitary_matrix_0dot1_0123456_7_qubits,
    MCRY_unitary_matrix_0dot1_01234567_8_qubits,
    MCRZ_unitary_matrix_pi_over_4_01_23_qubits,
    MCRZ_unitary_matrix_pi_over_4_10_23_qubits,
    MCRZ_unitary_matrix_1_over_4_02_13_qubits,
    MCRZ_unitary_matrix_1_over_4_20_34_qubits,
    MCRZ_unitary_matrix_negative1_over_4_12_04_qubits,
    MCRZ_unitary_matrix_negative1_over_4_53_01_qubits,
    MCRZ_unitary_matrix_1_over_3_012_34_qubits,
    MCRZ_unitary_matrix_1_over_3_01_234_qubits,
    MCRZ_unitary_matrix_pi_over_4_012_345_qubits,
    MCRZ_unitary_matrix_pi_over_4_01_2_qubits,
    MCRZ_unitary_matrix_pi_over_4_0_23_qubits,
    MCRZ_unitary_matrix_0dot1_012345_6_qubits,
    MCRZ_unitary_matrix_0dot1_0123456_7_qubits,
    MCRZ_unitary_matrix_0dot1_01234567_8_qubits,
    MCPhase_unitary_matrix_pi_over_4_01_23_qubits,
    MCPhase_unitary_matrix_pi_over_4_10_23_qubits,
    MCPhase_unitary_matrix_1_over_4_02_13_qubits,
    MCPhase_unitary_matrix_1_over_4_20_34_qubits,
    MCPhase_unitary_matrix_negative1_over_4_12_04_qubits,
    MCPhase_unitary_matrix_negative1_over_4_53_01_qubits,
    MCPhase_unitary_matrix_1_over_3_012_34_qubits,
    MCPhase_unitary_matrix_1_over_3_01_234_qubits,
    MCPhase_unitary_matrix_pi_over_4_012_345_qubits,
    MCPhase_unitary_matrix_pi_over_4_01_2_qubits,
    MCPhase_unitary_matrix_pi_over_4_0_23_qubits,
    MCXPow_unitary_matrix_1_over_4_0_shift_01_23_qubits,
    MCXPow_unitary_matrix_1_over_4_0_shift_10_23_qubits,
    MCXPow_unitary_matrix_1_over_4_0_shift_02_13_qubits,
    MCXPow_unitary_matrix_1_over_4_0_shift_20_34_qubits,
    MCXPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits,
    MCXPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits,
    MCXPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits,
    MCXPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits,
    MCYPow_unitary_matrix_1_over_4_0_shift_01_23_qubits,
    MCYPow_unitary_matrix_1_over_4_0_shift_10_23_qubits,
    MCYPow_unitary_matrix_1_over_4_0_shift_02_13_qubits,
    MCYPow_unitary_matrix_1_over_4_0_shift_20_34_qubits,
    MCYPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits,
    MCYPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits,
    MCYPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits,
    MCYPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits,
    MCZPow_unitary_matrix_1_over_4_0_shift_01_23_qubits,
    MCZPow_unitary_matrix_1_over_4_0_shift_10_23_qubits,
    MCZPow_unitary_matrix_1_over_4_0_shift_02_13_qubits,
    MCZPow_unitary_matrix_1_over_4_0_shift_20_34_qubits,
    MCZPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits,
    MCZPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits,
    MCZPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits,
    MCZPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits,
    MCRXX_unitary_matrix_pi_over_4_01_23_qubits,
    MCRXX_unitary_matrix_pi_over_4_10_23_qubits,
    MCRXX_unitary_matrix_1_over_4_02_13_qubits,
    MCRXX_unitary_matrix_1_over_4_20_34_qubits,
    MCRXX_unitary_matrix_negative1_over_4_12_04_qubits,
    MCRXX_unitary_matrix_negative1_over_4_53_01_qubits,
    MCRXX_unitary_matrix_1_over_3_012_34_qubits,
    MCRXX_unitary_matrix_1_over_3_0_23_qubits,
    MCRYY_unitary_matrix_pi_over_4_01_23_qubits,
    MCRYY_unitary_matrix_pi_over_4_10_23_qubits,
    MCRYY_unitary_matrix_1_over_4_02_13_qubits,
    MCRYY_unitary_matrix_1_over_4_20_34_qubits,
    MCRYY_unitary_matrix_negative1_over_4_12_04_qubits,
    MCRYY_unitary_matrix_negative1_over_4_53_01_qubits,
    MCRYY_unitary_matrix_1_over_3_012_34_qubits,
    MCRYY_unitary_matrix_1_over_3_0_23_qubits,
    MCRZZ_unitary_matrix_pi_over_4_01_23_qubits,
    MCRZZ_unitary_matrix_pi_over_4_10_23_qubits,
    MCRZZ_unitary_matrix_1_over_4_02_13_qubits,
    MCRZZ_unitary_matrix_1_over_4_20_34_qubits,
    MCRZZ_unitary_matrix_negative1_over_4_12_04_qubits,
    MCRZZ_unitary_matrix_negative1_over_4_53_01_qubits,
    MCRZZ_unitary_matrix_1_over_3_012_34_qubits,
    MCRZZ_unitary_matrix_1_over_3_0_23_qubits,
    MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_01_23_qubits,
    MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_10_23_qubits,
    MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_02_13_qubits,
    MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_20_34_qubits,
    MCU3_unitary_matrix_negative1_over_2_negative1_over_3_negative1_over_4_12_04_qubits,
    MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_123_45_qubits,
    MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_0_12_qubits,
    MCSWAP_unitary_matrix_01_23_qubits,
    MCSWAP_unitary_matrix_10_23_qubits,
    MCSWAP_unitary_matrix_02_13_qubits,
    MCSWAP_unitary_matrix_20_34_qubits,
    MCSWAP_unitary_matrix_123_45_qubits,
    MCSWAP_unitary_matrix_0_23_qubits
)


class TestTKETCircuit(Template):
    """ `tests.circuit.TestTKETCircuit` is the tester class for `qickit.circuit.TKETCircuit` class.
    """
    def test_init(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        TKETCircuit(1)

    def test_num_qubits_value(self) -> None:
        # Ensure the error is raised when the number of qubits is less than or equal to 0
        with pytest.raises(ValueError):
            TKETCircuit(0)

        with pytest.raises(ValueError):
            TKETCircuit(-1)

    def test_num_qubits_type(self) -> None:
        # Ensure the error is raised when the number of qubits is not an integer
        with pytest.raises(TypeError):
            TKETCircuit(1.0) # type: ignore

    def test_single_qubit_gate_from_range(self) -> None:
        """ Test the single qubit gate when indices are passed as a range instance.
        """
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(3)

        # Define the qubit indices as a range of ints
        qubit_indices = range(3)

        # Apply the Pauli-X gate
        circuit.X(qubit_indices)

        # Define the checker
        checker_circuit = TKETCircuit(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    def test_single_qubit_gate_from_tuple(self) -> None:
        """ Test the single qubit gate when indices are passed as a tuple instance.
        """
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(3)

        # Define the qubit indices as a tuple of ints
        qubit_indices = (0, 1, 2)

        # Apply the Pauli-X gate
        circuit.X(qubit_indices)

        # Define the checker
        checker_circuit = TKETCircuit(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    def test_single_qubit_gate_from_ndarray(self) -> None:
        """ Test the single qubit gate when indices are passed as a numpy.ndarray instance.
        """
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(3)

        # Define the qubit indices as a ndarray of ints
        qubit_indices = np.array([0, 1, 2])

        # Apply the Pauli-X gate
        circuit.X(qubit_indices) # type: ignore

        # Define the checker
        checker_circuit = TKETCircuit(3)
        checker_circuit.X([0, 1, 2])

        assert circuit == checker_circuit

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, np.eye(2**1)],
        [2, 1, np.eye(2**2)],
        [3, 2, np.eye(2**3)],
        [1, [0], np.eye(2**1)],
        [3, [0, 1], np.eye(2**3)],
        [3, [0, 2], np.eye(2**3)],
        [3, [1, 2], np.eye(2**3)],
        [3, [0, 1, 2], np.eye(2**3)]
    ])
    def test_Identity(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Identity gate
        circuit.Identity(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, X_unitary_matrix],
        [2, 1, np.kron(X_unitary_matrix, np.eye(2))],
        [3, 2, np.kron(X_unitary_matrix, np.eye(4))],
        [1, [0], X_unitary_matrix],
        [3, [0, 1], np.kron(np.eye(2), np.kron(X_unitary_matrix, X_unitary_matrix))],
        [3, [0, 2], np.kron(X_unitary_matrix, np.kron(np.eye(2), X_unitary_matrix))],
        [3, [1, 2], np.kron(X_unitary_matrix, np.kron(X_unitary_matrix, np.eye(2)))],
        [3, [0, 1, 2], np.kron(X_unitary_matrix, np.kron(X_unitary_matrix, X_unitary_matrix))]
    ])
    def test_X(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Pauli X gate
        circuit.X(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, Y_unitary_matrix],
        [2, 1, np.kron(Y_unitary_matrix, np.eye(2))],
        [3, 2, np.kron(Y_unitary_matrix, np.eye(4))],
        [1, [0], Y_unitary_matrix],
        [3, [0, 1], np.kron(np.eye(2), np.kron(Y_unitary_matrix, Y_unitary_matrix))],
        [3, [0, 2], np.kron(Y_unitary_matrix, np.kron(np.eye(2), Y_unitary_matrix))],
        [3, [1, 2], np.kron(Y_unitary_matrix, np.kron(Y_unitary_matrix, np.eye(2)))],
        [3, [0, 1, 2], np.kron(Y_unitary_matrix, np.kron(Y_unitary_matrix, Y_unitary_matrix))]
    ])
    def test_Y(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Pauli Y gate
        circuit.Y(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, Z_unitary_matrix],
        [2, 1, np.kron(Z_unitary_matrix, np.eye(2))],
        [3, 2, np.kron(Z_unitary_matrix, np.eye(4))],
        [1, [0], Z_unitary_matrix],
        [3, [0, 1], np.kron(np.eye(2), np.kron(Z_unitary_matrix, Z_unitary_matrix))],
        [3, [0, 2], np.kron(Z_unitary_matrix, np.kron(np.eye(2), Z_unitary_matrix))],
        [3, [1, 2], np.kron(Z_unitary_matrix, np.kron(Z_unitary_matrix, np.eye(2)))],
        [3, [0, 1, 2], np.kron(Z_unitary_matrix, np.kron(Z_unitary_matrix, Z_unitary_matrix))]
    ])
    def test_Z(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Pauli Z gate
        circuit.Z(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, H_unitary_matrix],
        [2, 1, np.kron(H_unitary_matrix, np.eye(2))],
        [3, 2, np.kron(H_unitary_matrix, np.eye(4))],
        [1, [0], H_unitary_matrix],
        [3, [0, 1], np.kron(np.eye(2), np.kron(H_unitary_matrix, H_unitary_matrix))],
        [3, [0, 2], np.kron(H_unitary_matrix, np.kron(np.eye(2), H_unitary_matrix))],
        [3, [1, 2], np.kron(H_unitary_matrix, np.kron(H_unitary_matrix, np.eye(2)))],
        [3, [0, 1, 2], np.kron(H_unitary_matrix, np.kron(H_unitary_matrix, H_unitary_matrix))]
    ])
    def test_H(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Hadamard gate
        circuit.H(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, S_unitary_matrix],
        [2, 1, np.kron(S_unitary_matrix, np.eye(2))],
        [3, 2, np.kron(S_unitary_matrix, np.eye(4))],
        [1, [0], S_unitary_matrix],
        [3, [0, 1], np.kron(np.eye(2), np.kron(S_unitary_matrix, S_unitary_matrix))],
        [3, [0, 2], np.kron(S_unitary_matrix, np.kron(np.eye(2), S_unitary_matrix))],
        [3, [1, 2], np.kron(S_unitary_matrix, np.kron(S_unitary_matrix, np.eye(2)))],
        [3, [0, 1, 2], np.kron(S_unitary_matrix, np.kron(S_unitary_matrix, S_unitary_matrix))]
    ])
    def test_S(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Clifford S gate
        circuit.S(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, S_unitary_matrix.conj().T],
        [2, 1, np.kron(S_unitary_matrix.conj().T, np.eye(2))],
        [3, 2, np.kron(S_unitary_matrix.conj().T, np.eye(4))],
        [1, [0], S_unitary_matrix.conj().T],
        [3, [0, 1], np.kron(np.eye(2), np.kron(S_unitary_matrix.conj().T, S_unitary_matrix.conj().T))],
        [3, [0, 2], np.kron(S_unitary_matrix.conj().T, np.kron(np.eye(2), S_unitary_matrix.conj().T))],
        [3, [1, 2], np.kron(S_unitary_matrix.conj().T, np.kron(S_unitary_matrix.conj().T, np.eye(2)))],
        [3, [0, 1, 2], np.kron(S_unitary_matrix.conj().T, np.kron(S_unitary_matrix.conj().T, S_unitary_matrix.conj().T))]
    ])
    def test_Sdg(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Clifford S dagger gate
        circuit.Sdg(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, T_unitary_matrix],
        [2, 1, np.kron(T_unitary_matrix, np.eye(2))],
        [3, 2, np.kron(T_unitary_matrix, np.eye(4))],
        [1, [0], T_unitary_matrix],
        [3, [0, 1], np.kron(np.eye(2), np.kron(T_unitary_matrix, T_unitary_matrix))],
        [3, [0, 2], np.kron(T_unitary_matrix, np.kron(np.eye(2), T_unitary_matrix))],
        [3, [1, 2], np.kron(T_unitary_matrix, np.kron(T_unitary_matrix, np.eye(2)))],
        [3, [0, 1, 2], np.kron(T_unitary_matrix, np.kron(T_unitary_matrix, T_unitary_matrix))]
    ])
    def test_T(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Clifford T gate
        circuit.T(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, expected", [
        [1, 0, T_unitary_matrix.conj().T],
        [2, 1, np.kron(T_unitary_matrix.conj().T, np.eye(2))],
        [3, 2, np.kron(T_unitary_matrix.conj().T, np.eye(4))],
        [1, [0], T_unitary_matrix.conj().T],
        [3, [0, 1], np.kron(np.eye(2), np.kron(T_unitary_matrix.conj().T, T_unitary_matrix.conj().T))],
        [3, [0, 2], np.kron(T_unitary_matrix.conj().T, np.kron(np.eye(2), T_unitary_matrix.conj().T))],
        [3, [1, 2], np.kron(T_unitary_matrix.conj().T, np.kron(T_unitary_matrix.conj().T, np.eye(2)))],
        [3, [0, 1, 2], np.kron(T_unitary_matrix.conj().T, np.kron(T_unitary_matrix.conj().T, T_unitary_matrix.conj().T))]
    ])
    def test_Tdg(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Clifford T dagger gate
        circuit.Tdg(qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, angle, expected", [
        [1, 0, np.pi/4, RX(np.pi/4).matrix],
        [1, 0, 1/3, RX(1/3).matrix],
        [1, 0, -1/4, RX(-1/4).matrix],
        [2, 1, np.pi/4, np.kron(RX(np.pi/4).matrix, np.eye(2))],
        [3, 2, 1/3, np.kron(RX(1/3).matrix, np.eye(4))],
        [3, [0, 1], np.pi/4, np.kron(np.eye(2), np.kron(RX(np.pi/4).matrix, RX(np.pi/4).matrix))]
    ])
    def test_RX(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the RX gate
        circuit.RX(angle, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, angle, expected", [
        [1, 0, np.pi/4, RY(np.pi/4).matrix],
        [1, 0, 1/3, RY(1/3).matrix],
        [1, 0, -1/4, RY(-1/4).matrix],
        [2, 1, np.pi/4, np.kron(RY(np.pi/4).matrix, np.eye(2))],
        [3, 2, 1/3, np.kron(RY(1/3).matrix, np.eye(4))],
        [3, [0, 1], np.pi/4, np.kron(np.eye(2), np.kron(RY(np.pi/4).matrix, RY(np.pi/4).matrix))]
    ])
    def test_RY(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the RY gate
        circuit.RY(angle, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, angle, expected", [
        [1, 0, np.pi/4, RZ(np.pi/4).matrix],
        [1, 0, 1/3, RZ(1/3).matrix],
        [1, 0, -1/4, RZ(-1/4).matrix],
        [2, 1, np.pi/4, np.kron(RZ(np.pi/4).matrix, np.eye(2))],
        [3, 2, 1/3, np.kron(RZ(1/3).matrix, np.eye(4))],
        [3, [0, 1], np.pi/4, np.kron(np.eye(2), np.kron(RZ(np.pi/4).matrix, RZ(np.pi/4).matrix))]
    ])
    def test_RZ(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the RZ gate
        circuit.RZ(angle, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, angle, expected", [
        [1, 0, np.pi/4, Phase(np.pi/4).matrix],
        [1, 0, 1/3, Phase(1/3).matrix],
        [1, 0, -1/4, Phase(-1/4).matrix],
        [2, 1, np.pi/4, np.kron(Phase(np.pi/4).matrix, np.eye(2))],
        [3, 2, 1/3, np.kron(Phase(1/3).matrix, np.eye(4))],
        [3, [0, 1], np.pi/4, np.kron(np.eye(2), np.kron(Phase(np.pi/4).matrix, Phase(np.pi/4).matrix))]
    ])
    def test_Phase(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the Phase gate
        circuit.Phase(angle, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, power, global_shift, expected", [
        [1, 0, 1/4, 0, XPow_unitary_matrix],
        [2, 1, 1/4, 0, np.kron(XPow_unitary_matrix, np.eye(2))],
        [3, 2, 1/4, 0, np.kron(XPow_unitary_matrix, np.eye(4))],
        [1, 0, 1/4, 1/3, XPow_global_shift_unitary_matrix],
        [2, 1, 1/4, 1/3, np.kron(XPow_global_shift_unitary_matrix, np.eye(2))],
        [3, 2, 1/4, 1/3, np.kron(XPow_global_shift_unitary_matrix, np.eye(4))],
        [3, [0, 1], 1/4, 0, np.kron(np.eye(2), np.kron(XPow_unitary_matrix, XPow_unitary_matrix))],
        [3, [0, 2], 1/4, 1/3, np.kron(XPow_global_shift_unitary_matrix, np.kron(np.eye(2), XPow_global_shift_unitary_matrix))]
    ])
    def test_XPow(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the XPow gate
        circuit.XPow(power, global_shift, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, power, global_shift, expected", [
        [1, 0, 1/4, 0, YPow_unitary_matrix],
        [2, 1, 1/4, 0, np.kron(YPow_unitary_matrix, np.eye(2))],
        [3, 2, 1/4, 0, np.kron(YPow_unitary_matrix, np.eye(4))],
        [1, 0, 1/4, 1/3, YPow_global_shift_unitary_matrix],
        [2, 1, 1/4, 1/3, np.kron(YPow_global_shift_unitary_matrix, np.eye(2))],
        [3, 2, 1/4, 1/3, np.kron(YPow_global_shift_unitary_matrix, np.eye(4))],
        [3, [0, 1], 1/4, 0, np.kron(np.eye(2), np.kron(YPow_unitary_matrix, YPow_unitary_matrix))],
        [3, [0, 2], 1/4, 1/3, np.kron(YPow_global_shift_unitary_matrix, np.kron(np.eye(2), YPow_global_shift_unitary_matrix))]
    ])
    def test_YPow(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the YPow gate
        circuit.YPow(power, global_shift, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, power, global_shift, expected", [
        [1, 0, 1/4, 0, ZPow_unitary_matrix],
        [2, 1, 1/4, 0, np.kron(ZPow_unitary_matrix, np.eye(2))],
        [3, 2, 1/4, 0, np.kron(ZPow_unitary_matrix, np.eye(4))],
        [1, 0, 1/4, 1/3, ZPow_global_shift_unitary_matrix],
        [2, 1, 1/4, 1/3, np.kron(ZPow_global_shift_unitary_matrix, np.eye(2))],
        [3, 2, 1/4, 1/3, np.kron(ZPow_global_shift_unitary_matrix, np.eye(4))],
        [3, [0, 1], 1/4, 0, np.kron(np.eye(2), np.kron(ZPow_unitary_matrix, ZPow_unitary_matrix))],
        [3, [0, 2], 1/4, 1/3, np.kron(ZPow_global_shift_unitary_matrix, np.kron(np.eye(2), ZPow_global_shift_unitary_matrix))]
    ])
    def test_ZPow(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the ZPow gate
        circuit.ZPow(power, global_shift, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, first_qubit_index, second_qubit_index, angle, expected", [
        [2, 0, 1, np.pi/4, RXX_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, RXX_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, RXX_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, RXX_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, RXX_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_RXX(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the RXX gate
        circuit.RXX(angle, first_qubit_index, second_qubit_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, first_qubit_index, second_qubit_index, angle, expected", [
        [2, 0, 1, np.pi/4, RYY_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, RYY_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, RYY_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, RYY_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, RYY_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_RYY(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the RYY gate
        circuit.RYY(angle, first_qubit_index, second_qubit_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, first_qubit_index, second_qubit_index, angle, expected", [
        [2, 0, 1, np.pi/4, RZZ_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, RZZ_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, RZZ_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, RZZ_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, RZZ_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_RZZ(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the RZZ gate
        circuit.RZZ(angle, first_qubit_index, second_qubit_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, qubit_indices, angles, expected", [
        [1, 0, (np.pi/2, np.pi/3, np.pi/4), U3(np.pi/2, np.pi/3, np.pi/4).matrix],
        [2, 1, (np.pi/2, -np.pi/3, np.pi/4), np.kron(U3(np.pi/2, -np.pi/3, np.pi/4).matrix, np.eye(2))],
        [3, 2, (np.pi/2, np.pi/3, -np.pi/4), np.kron(U3(np.pi/2, np.pi/3, -np.pi/4).matrix, np.eye(4))],
        [1, 0, (1/3, 1/4, 1/5), U3(1/3, 1/4, 1/5).matrix],
        [3, [0, 2], (np.pi/2, np.pi/3, np.pi/4), np.kron(
            U3(np.pi/2, np.pi/3, np.pi/4).matrix,
            np.kron(
                np.eye(2),
                U3(np.pi/2, np.pi/3, np.pi/4).matrix
            )
        )],
        [3, [0, 1], (np.pi/2, np.pi/3, np.pi/4), np.kron(
                np.eye(2),
                np.kron(
                    U3(np.pi/2, np.pi/3, np.pi/4).matrix,
                    U3(np.pi/2, np.pi/3, np.pi/4).matrix
                )
            )
        ]
    ])
    def test_U3(
            self,
            num_qubits: int,
            qubit_indices: int | list[int],
            angles: tuple[float, float, float],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the U3 gate
        circuit.U3(angles, qubit_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, first_qubit_index, second_qubit_index, expected", [
        [2, 0, 1, SWAP_unitary_matrix_01qubits],
        [2, 1, 0, SWAP_unitary_matrix_10qubits],
        [3, 0, 2, SWAP_unitary_matrix_02qubits],
        [3, 2, 0, SWAP_unitary_matrix_20qubits],
        [3, 1, 2, SWAP_unitary_matrix_12qubits]

    ])
    def test_SWAP(
            self,
            num_qubits: int,
            first_qubit_index: int,
            second_qubit_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the SWAP gate
        circuit.SWAP(first_qubit_index, second_qubit_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CX_unitary_matrix_01qubits],
        [2, 1, 0, CX_unitary_matrix_10qubits],
        [3, 0, 2, CX_unitary_matrix_02qubits],
        [3, 2, 0, CX_unitary_matrix_20qubits],
        [3, 1, 2, CX_unitary_matrix_12qubits]

    ])
    def test_CX(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CX gate
        circuit.CX(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CY_unitary_matrix_01qubits],
        [2, 1, 0, CY_unitary_matrix_10qubits],
        [3, 0, 2, CY_unitary_matrix_02qubits],
        [3, 2, 0, CY_unitary_matrix_20qubits],
        [3, 1, 2, CY_unitary_matrix_12qubits]

    ])
    def test_CY(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CY gate
        circuit.CY(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CZ_unitary_matrix_01qubits],
        [2, 1, 0, CZ_unitary_matrix_10qubits],
        [3, 0, 2, CZ_unitary_matrix_02qubits],
        [3, 2, 0, CZ_unitary_matrix_20qubits],
        [3, 1, 2, CZ_unitary_matrix_12qubits]

    ])
    def test_CZ(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CZ gate
        circuit.CZ(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CH_unitary_matrix_01qubits],
        [2, 1, 0, CH_unitary_matrix_10qubits],
        [3, 0, 2, CH_unitary_matrix_02qubits],
        [3, 2, 0, CH_unitary_matrix_20qubits],
        [3, 1, 2, CH_unitary_matrix_12qubits]

    ])
    def test_CH(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CH gate
        circuit.CH(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CS_unitary_matrix_01qubits],
        [2, 1, 0, CS_unitary_matrix_10qubits],
        [3, 0, 2, CS_unitary_matrix_02qubits],
        [3, 2, 0, CS_unitary_matrix_20qubits],
        [3, 1, 2, CS_unitary_matrix_12qubits]

    ])
    def test_CS(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CS gate
        circuit.CS(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CSdg_unitary_matrix_01qubits],
        [2, 1, 0, CSdg_unitary_matrix_10qubits],
        [3, 0, 2, CSdg_unitary_matrix_02qubits],
        [3, 2, 0, CSdg_unitary_matrix_20qubits],
        [3, 1, 2, CSdg_unitary_matrix_12qubits]

    ])
    def test_CSdg(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CSdg gate
        circuit.CSdg(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CT_unitary_matrix_01qubits],
        [2, 1, 0, CT_unitary_matrix_10qubits],
        [3, 0, 2, CT_unitary_matrix_02qubits],
        [3, 2, 0, CT_unitary_matrix_20qubits],
        [3, 1, 2, CT_unitary_matrix_12qubits]

    ])
    def test_CT(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CT gate
        circuit.CT(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, expected", [
        [2, 0, 1, CTdg_unitary_matrix_01qubits],
        [2, 1, 0, CTdg_unitary_matrix_10qubits],
        [3, 0, 2, CTdg_unitary_matrix_02qubits],
        [3, 2, 0, CTdg_unitary_matrix_20qubits],
        [3, 1, 2, CTdg_unitary_matrix_12qubits]

    ])
    def test_CTdg(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CTdg gate
        circuit.CTdg(control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, angle, expected", [
        [2, 0, 1, np.pi/4, CRX_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, CRX_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, CRX_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, CRX_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, CRX_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_CRX(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CRX gate
        circuit.CRX(angle, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, angle, expected", [
        [2, 0, 1, np.pi/4, CRY_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, CRY_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, CRY_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, CRY_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, CRY_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_CRY(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CRY gate
        circuit.CRY(angle, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, angle, expected", [
        [2, 0, 1, np.pi/4, CRZ_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, CRZ_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, CRZ_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, CRZ_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, CRZ_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_CRZ(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CRZ gate
        circuit.CRZ(angle, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, angle, expected", [
        [2, 0, 1, np.pi/4, CPhase_unitary_matrix_pi_over_4_01qubits],
        [2, 1, 0, np.pi/4, CPhase_unitary_matrix_pi_over_4_10qubits],
        [3, 0, 2, 1/4, CPhase_unitary_matrix_1_over_4_02qubits],
        [3, 2, 0, 1/4, CPhase_unitary_matrix_1_over_4_20qubits],
        [3, 1, 2, np.pi/4, CPhase_unitary_matrix_pi_over_4_12qubits]
    ])
    def test_CPhase(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CPhase gate
        circuit.CPhase(angle, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, power, global_shift, expected", [
        [2, 0, 1, 1/4, 0, CXPow_unitary_matrix_1_over_4_0_shift_01qubits],
        [2, 1, 0, 1/4, 0, CXPow_unitary_matrix_1_over_4_0_shift_10qubits],
        [3, 0, 2, 1/4, 1/3, CXPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits],
        [3, 2, 0, 1/4, 1/3, CXPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits],
        [3, 1, 2, 1/4, 0, CXPow_unitary_matrix_negative1_over_4_0_shift_12qubits]
    ])
    def test_CXPow(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CXPow gate
        circuit.CXPow(power, global_shift, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, power, global_shift, expected", [
        [2, 0, 1, 1/4, 0, CYPow_unitary_matrix_1_over_4_0_shift_01qubits],
        [2, 1, 0, 1/4, 0, CYPow_unitary_matrix_1_over_4_0_shift_10qubits],
        [3, 0, 2, 1/4, 1/3, CYPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits],
        [3, 2, 0, 1/4, 1/3, CYPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits],
        [3, 1, 2, 1/4, 0, CYPow_unitary_matrix_negative1_over_4_0_shift_12qubits]
    ])
    def test_CYPow(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CYPow gate
        circuit.CYPow(power, global_shift, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, power, global_shift, expected", [
        [2, 0, 1, 1/4, 0, CZPow_unitary_matrix_1_over_4_0_shift_01qubits],
        [2, 1, 0, 1/4, 0, CZPow_unitary_matrix_1_over_4_0_shift_10qubits],
        [3, 0, 2, 1/4, 1/3, CZPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits],
        [3, 2, 0, 1/4, 1/3, CZPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits],
        [3, 1, 2, 1/4, 0, CZPow_unitary_matrix_negative1_over_4_0_shift_12qubits]
    ])
    def test_CZPow(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CZPow gate
        circuit.CZPow(power, global_shift, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, first_target_index, second_target_index, angle, expected", [
        [3, 0, 1, 2, np.pi/4, CRXX_unitary_matrix_pi_over_4_012qubits],
        [3, 1, 0, 2, np.pi/4, CRXX_unitary_matrix_pi_over_4_102qubits],
        [4, 1, 2, 3, 1/4, CRXX_unitary_matrix_1_over_4_123qubits],
        [4, 2, 1, 3, 1/4, CRXX_unitary_matrix_1_over_4_213qubits],
        [4, 0, 2, 3, np.pi/4, CRXX_unitary_matrix_pi_over_4_023qubits]
    ])
    def test_CRXX(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CRXX gate
        circuit.CRXX(angle, control_index, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, first_target_index, second_target_index, angle, expected", [
        [3, 0, 1, 2, np.pi/4, CRYY_unitary_matrix_pi_over_4_012qubits],
        [3, 1, 0, 2, np.pi/4, CRYY_unitary_matrix_pi_over_4_102qubits],
        [4, 1, 2, 3, 1/4, CRYY_unitary_matrix_1_over_4_123qubits],
        [4, 2, 1, 3, 1/4, CRYY_unitary_matrix_1_over_4_213qubits],
        [4, 0, 2, 3, np.pi/4, CRYY_unitary_matrix_pi_over_4_023qubits]
    ])
    def test_CRYY(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CRYY gate
        circuit.CRYY(angle, control_index, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, first_target_index, second_target_index, angle, expected", [
        [3, 0, 1, 2, np.pi/4, CRZZ_unitary_matrix_pi_over_4_012qubits],
        [3, 1, 0, 2, np.pi/4, CRZZ_unitary_matrix_pi_over_4_102qubits],
        [4, 1, 2, 3, 1/4, CRZZ_unitary_matrix_1_over_4_123qubits],
        [4, 2, 1, 3, 1/4, CRZZ_unitary_matrix_1_over_4_213qubits],
        [4, 0, 2, 3, np.pi/4, CRZZ_unitary_matrix_pi_over_4_023qubits]
    ])
    def test_CRZZ(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CRZZ gate
        circuit.CRZZ(angle, control_index, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, target_index, angles, expected", [
        [2, 0, 1, (np.pi/2, np.pi/3, np.pi/4), CU3_unitary_matrix_pi2_pi3_pi4_01qubits],
        [2, 1, 0, (np.pi/2, np.pi/3, np.pi/4), CU3_unitary_matrix_pi2_pi3_pi4_10qubits],
        [3, 0, 2, (np.pi/2, -np.pi/3, np.pi/4), CU3_unitary_matrix_pi2_pi3_pi4_02qubits],
        [3, 2, 0, (np.pi/2, np.pi/3, -np.pi/4), CU3_unitary_matrix_pi2_pi3_pi4_20qubits],
        [3, 1, 2, (-np.pi/2, np.pi/3, np.pi/4), CU3_unitary_matrix_pi2_pi3_pi4_12qubits]
    ])
    def test_CU3(
            self,
            num_qubits: int,
            control_index: int,
            target_index: int,
            angles: tuple[float, float, float],
            expected
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CU3 gate
        circuit.CU3(angles, control_index, target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_index, first_target_index, second_target_index, expected", [
        [3, 0, 1, 2, CSWAP_unitary_matrix_012qubits],
        [3, 1, 0, 2, CSWAP_unitary_matrix_102qubits],
        [4, 1, 2, 3, CSWAP_unitary_matrix_123qubits],
        [4, 2, 1, 3, CSWAP_unitary_matrix_213qubits],
        [4, 0, 2, 3, CSWAP_unitary_matrix_023qubits]
    ])
    def test_CSWAP(
            self,
            num_qubits: int,
            control_index: int,
            first_target_index: int,
            second_target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the CSWAP gate
        circuit.CSWAP(control_index, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCX_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCX_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCX_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCX_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCX_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCX_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCX_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCX_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCX_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCX_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCX_unitary_matrix_0_23_qubits]
    ])
    def test_MCX(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCX gate
        circuit.MCX(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCY_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCY_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCY_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCY_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCY_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCY_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCY_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCY_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCY_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCY_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCY_unitary_matrix_0_23_qubits]
    ])
    def test_MCY(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCY gate
        circuit.MCY(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCZ_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCZ_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCZ_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCZ_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCZ_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCZ_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCZ_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCZ_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCZ_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCZ_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCZ_unitary_matrix_0_23_qubits]
    ])
    def test_MCZ(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCZ gate
        circuit.MCZ(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCH_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCH_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCH_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCH_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCH_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCH_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCH_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCH_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCH_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCH_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCH_unitary_matrix_0_23_qubits]
    ])
    def test_MCH(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCH gate
        circuit.MCH(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCS_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCS_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCS_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCS_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCS_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCS_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCS_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCS_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCS_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCS_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCS_unitary_matrix_0_23_qubits]
    ])
    def test_MCS(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCS gate
        circuit.MCS(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCSdg_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCSdg_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCSdg_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCSdg_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCSdg_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCSdg_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCSdg_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCSdg_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCSdg_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCSdg_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCSdg_unitary_matrix_0_23_qubits]
    ])
    def test_MCSdg(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCSdg gate
        circuit.MCSdg(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCT_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCT_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCT_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCT_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCT_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCT_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCT_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCT_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCT_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCT_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCT_unitary_matrix_0_23_qubits]
    ])
    def test_MCT(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCT gate
        circuit.MCT(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, expected", [
        [4, [0, 1], [2, 3], MCTdg_unitary_matrix_01_23_qubits],
        [4, [1, 0], [2, 3], MCTdg_unitary_matrix_10_23_qubits],
        [5, [0, 2], [1, 3], MCTdg_unitary_matrix_02_13_qubits],
        [5, [2, 0], [3, 4], MCTdg_unitary_matrix_20_34_qubits],
        [5, [1, 2], [0, 4], MCTdg_unitary_matrix_12_04_qubits],
        [6, [5, 3], [0, 1], MCTdg_unitary_matrix_53_01_qubits],
        [5, [0, 1, 2], [3, 4], MCTdg_unitary_matrix_012_34_qubits],
        [5, [0, 1], [2, 3, 4], MCTdg_unitary_matrix_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], MCTdg_unitary_matrix_012_345_qubits],
        [4, [0, 1], [2], MCTdg_unitary_matrix_01_2_qubits],
        [4, [0], [2, 3], MCTdg_unitary_matrix_0_23_qubits]
    ])
    def test_MCTdg(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCTdg gate
        circuit.MCTdg(control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, angle, expected", [
        [4, [0, 1], [2, 3], np.pi/4, MCRX_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], [2, 3], np.pi/4, MCRX_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, MCRX_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, MCRX_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, MCRX_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], [0, 1], -1/4, MCRX_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], [3, 4], 1/3, MCRX_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0, 1], [2, 3, 4], 1/3, MCRX_unitary_matrix_1_over_3_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], np.pi/4, MCRX_unitary_matrix_pi_over_4_012_345_qubits],
        [4, [0, 1], [2], np.pi/4, MCRX_unitary_matrix_pi_over_4_01_2_qubits],
        [4, [0], [2, 3], np.pi/4, MCRX_unitary_matrix_pi_over_4_0_23_qubits],
        [7, [0, 1, 2, 3, 4, 5], [6], 0.1, MCRX_unitary_matrix_0dot1_012345_6_qubits],
        [8, [0, 1, 2, 3, 4, 5, 6], [7], 0.1, MCRX_unitary_matrix_0dot1_0123456_7_qubits],
        [9, [0, 1, 2, 3, 4, 5, 6, 7], [8], 0.1, MCRX_unitary_matrix_0dot1_01234567_8_qubits]
    ])
    def test_MCRX(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCRX gate
        circuit.MCRX(angle, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, angle, expected", [
        [4, [0, 1], [2, 3], np.pi/4, MCRY_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], [2, 3], np.pi/4, MCRY_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, MCRY_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, MCRY_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, MCRY_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], [0, 1], -1/4, MCRY_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], [3, 4], 1/3, MCRY_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0, 1], [2, 3, 4], 1/3, MCRY_unitary_matrix_1_over_3_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], np.pi/4, MCRY_unitary_matrix_pi_over_4_012_345_qubits],
        [4, [0, 1], [2], np.pi/4, MCRY_unitary_matrix_pi_over_4_01_2_qubits],
        [4, [0], [2, 3], np.pi/4, MCRY_unitary_matrix_pi_over_4_0_23_qubits],
        [7, [0, 1, 2, 3, 4, 5], [6], 0.1, MCRY_unitary_matrix_0dot1_012345_6_qubits],
        [8, [0, 1, 2, 3, 4, 5, 6], [7], 0.1, MCRY_unitary_matrix_0dot1_0123456_7_qubits],
        [9, [0, 1, 2, 3, 4, 5, 6, 7], [8], 0.1, MCRY_unitary_matrix_0dot1_01234567_8_qubits]
    ])
    def test_MCRY(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCRY gate
        circuit.MCRY(angle, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, angle, expected", [
        [4, [0, 1], [2, 3], np.pi/4, MCRZ_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], [2, 3], np.pi/4, MCRZ_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, MCRZ_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, MCRZ_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, MCRZ_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], [0, 1], -1/4, MCRZ_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], [3, 4], 1/3, MCRZ_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0, 1], [2, 3, 4], 1/3, MCRZ_unitary_matrix_1_over_3_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], np.pi/4, MCRZ_unitary_matrix_pi_over_4_012_345_qubits],
        [4, [0, 1], [2], np.pi/4, MCRZ_unitary_matrix_pi_over_4_01_2_qubits],
        [4, [0], [2, 3], np.pi/4, MCRZ_unitary_matrix_pi_over_4_0_23_qubits],
        [7, [0, 1, 2, 3, 4, 5], [6], 0.1, MCRZ_unitary_matrix_0dot1_012345_6_qubits],
        [8, [0, 1, 2, 3, 4, 5, 6], [7], 0.1, MCRZ_unitary_matrix_0dot1_0123456_7_qubits],
        [9, [0, 1, 2, 3, 4, 5, 6, 7], [8], 0.1, MCRZ_unitary_matrix_0dot1_01234567_8_qubits]
    ])
    def test_MCRZ(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCRZ gate
        circuit.MCRZ(angle, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, angle, expected", [
        [4, [0, 1], [2, 3], np.pi/4, MCPhase_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], [2, 3], np.pi/4, MCPhase_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, MCPhase_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, MCPhase_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, MCPhase_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], [0, 1], -1/4, MCPhase_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], [3, 4], 1/3, MCPhase_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0, 1], [2, 3, 4], 1/3, MCPhase_unitary_matrix_1_over_3_01_234_qubits],
        [6, [0, 1, 2], [3, 4, 5], np.pi/4, MCPhase_unitary_matrix_pi_over_4_012_345_qubits],
        [4, [0, 1], [2], np.pi/4, MCPhase_unitary_matrix_pi_over_4_01_2_qubits],
        [4, [0], [2, 3], np.pi/4, MCPhase_unitary_matrix_pi_over_4_0_23_qubits]
    ])
    def test_MCPhase(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCPhase gate
        circuit.MCPhase(angle, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, power, global_shift, expected", [
        [4, [0, 1], [2, 3], 1/4, 0, MCXPow_unitary_matrix_1_over_4_0_shift_01_23_qubits],
        [4, [1, 0], [2, 3], 1/4, 0, MCXPow_unitary_matrix_1_over_4_0_shift_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, 0, MCXPow_unitary_matrix_1_over_4_0_shift_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, 0, MCXPow_unitary_matrix_1_over_4_0_shift_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, 0, MCXPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits],
        [4, [0, 1], [2, 3], 1/4, 1/3, MCXPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits],
        [4, [0], [2, 3], 1/4, 1/3, MCXPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits],
        [4, [0, 1], [2, 3], -1/4, -1/3, MCXPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits]
    ])
    def test_MCXPow(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCXPow gate
        circuit.MCXPow(power, global_shift, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, power, global_shift, expected", [
        [4, [0, 1], [2, 3], 1/4, 0, MCYPow_unitary_matrix_1_over_4_0_shift_01_23_qubits],
        [4, [1, 0], [2, 3], 1/4, 0, MCYPow_unitary_matrix_1_over_4_0_shift_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, 0, MCYPow_unitary_matrix_1_over_4_0_shift_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, 0, MCYPow_unitary_matrix_1_over_4_0_shift_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, 0, MCYPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits],
        [4, [0, 1], [2, 3], 1/4, 1/3, MCYPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits],
        [4, [0], [2, 3], 1/4, 1/3, MCYPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits],
        [4, [0, 1], [2, 3], -1/4, -1/3, MCYPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits]
    ])
    def test_MCYPow(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCYPow gate
        circuit.MCYPow(power, global_shift, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, power, global_shift, expected", [
        [4, [0, 1], [2, 3], 1/4, 0, MCZPow_unitary_matrix_1_over_4_0_shift_01_23_qubits],
        [4, [1, 0], [2, 3], 1/4, 0, MCZPow_unitary_matrix_1_over_4_0_shift_10_23_qubits],
        [5, [0, 2], [1, 3], 1/4, 0, MCZPow_unitary_matrix_1_over_4_0_shift_02_13_qubits],
        [5, [2, 0], [3, 4], 1/4, 0, MCZPow_unitary_matrix_1_over_4_0_shift_20_34_qubits],
        [5, [1, 2], [0, 4], -1/4, 0, MCZPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits],
        [4, [0, 1], [2, 3], 1/4, 1/3, MCZPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits],
        [4, [0], [2, 3], 1/4, 1/3, MCZPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits],
        [4, [0, 1], [2, 3], -1/4, -1/3, MCZPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits]
    ])
    def test_MCZPow(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            power: float,
            global_shift: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCZPow gate
        circuit.MCZPow(power, global_shift, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, first_target_index, second_target_index, angle, expected", [
        [4, [0, 1], 2, 3, np.pi/4, MCRXX_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], 2, 3, np.pi/4, MCRXX_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], 1, 3, 1/4, MCRXX_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], 3, 4, 1/4, MCRXX_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], 0, 4, -1/4, MCRXX_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], 0, 1, -1/4, MCRXX_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], 3, 4, 1/3, MCRXX_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0], 2, 3, 1/3, MCRXX_unitary_matrix_1_over_3_0_23_qubits]
    ])
    def test_MCRXX(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCRXX gate
        circuit.MCRXX(angle, control_indices, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, first_target_index, second_target_index, angle, expected", [
        [4, [0, 1], 2, 3, np.pi/4, MCRYY_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], 2, 3, np.pi/4, MCRYY_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], 1, 3, 1/4, MCRYY_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], 3, 4, 1/4, MCRYY_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], 0, 4, -1/4, MCRYY_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], 0, 1, -1/4, MCRYY_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], 3, 4, 1/3, MCRYY_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0], 2, 3, 1/3, MCRYY_unitary_matrix_1_over_3_0_23_qubits]
    ])
    def test_MCRYY(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCRYY gate
        circuit.MCRYY(angle, control_indices, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, first_target_index, second_target_index, angle, expected", [
        [4, [0, 1], 2, 3, np.pi/4, MCRZZ_unitary_matrix_pi_over_4_01_23_qubits],
        [4, [1, 0], 2, 3, np.pi/4, MCRZZ_unitary_matrix_pi_over_4_10_23_qubits],
        [5, [0, 2], 1, 3, 1/4, MCRZZ_unitary_matrix_1_over_4_02_13_qubits],
        [5, [2, 0], 3, 4, 1/4, MCRZZ_unitary_matrix_1_over_4_20_34_qubits],
        [5, [1, 2], 0, 4, -1/4, MCRZZ_unitary_matrix_negative1_over_4_12_04_qubits],
        [6, [5, 3], 0, 1, -1/4, MCRZZ_unitary_matrix_negative1_over_4_53_01_qubits],
        [5, [0, 1, 2], 3, 4, 1/3, MCRZZ_unitary_matrix_1_over_3_012_34_qubits],
        [5, [0], 2, 3, 1/3, MCRZZ_unitary_matrix_1_over_3_0_23_qubits]
    ])
    def test_MCRZZ(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            angle: float,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCRZZ gate
        circuit.MCRZZ(angle, control_indices, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, target_indices, angles, expected", [
        [4, [0, 1], [2, 3], (np.pi/2, np.pi/3, np.pi/4), MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_01_23_qubits],
        [4, [1, 0], [2, 3], (np.pi/2, np.pi/3, np.pi/4), MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_10_23_qubits],
        [5, [0, 2], [1, 3], (1/2, 1/3, 1/4), MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_02_13_qubits],
        [5, [2, 0], [3, 4], (1/2, 1/3, 1/4), MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_20_34_qubits],
        [5, [1, 2], [0, 4], (-1/2, -1/3, -1/4), MCU3_unitary_matrix_negative1_over_2_negative1_over_3_negative1_over_4_12_04_qubits],
        [6, [1, 2, 3], [4, 5], (np.pi/2, np.pi/3, np.pi/4), MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_123_45_qubits],
        [4, [0], [1, 2], (np.pi/2, np.pi/3, np.pi/4), MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_0_12_qubits]
    ])
    def test_MCU3(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            target_indices: int | list[int],
            angles: tuple[float, float, float],
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCU3 gate
        circuit.MCU3(angles, control_indices, target_indices)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("num_qubits, control_indices, first_target_index, second_target_index, expected", [
        [4, [0, 1], 2, 3, MCSWAP_unitary_matrix_01_23_qubits],
        [4, [1, 0], 2, 3, MCSWAP_unitary_matrix_10_23_qubits],
        [5, [0, 2], 1, 3, MCSWAP_unitary_matrix_02_13_qubits],
        [5, [2, 0], 3, 4, MCSWAP_unitary_matrix_20_34_qubits],
        [6, [1, 2, 3], 4, 5, MCSWAP_unitary_matrix_123_45_qubits],
        [4, [0], 2, 3, MCSWAP_unitary_matrix_0_23_qubits]
    ])
    def test_MCSWAP(
            self,
            num_qubits: int,
            control_indices: int | list[int],
            first_target_index: int,
            second_target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(num_qubits)

        # Apply the MCSWAP gate
        circuit.MCSWAP(control_indices, first_target_index, second_target_index)

        assert_almost_equal(circuit.get_unitary(), expected, 8)

    def test_GlobalPhase(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(1)

        # Generate a random phase angle
        phase = np.random.rand()

        # Apply the global phase gate
        circuit.GlobalPhase(phase)

        # Ensure the global phase is correct
        assert_almost_equal(circuit.get_unitary(), np.exp(phase * 1j) * np.eye(2), 8)

    def test_single_measurement(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(1)

        # Ensure no qubits are measured initially
        assert len(circuit.measured_qubits) == 0

        # Apply the measurement gate
        circuit.measure(0)

        # Ensure only the first qubit is measured
        assert circuit.measured_qubits == {0}

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(1, 1)
        tket_circuit.Measure(0, 0)

        assert circuit.circuit == tket_circuit

    def test_multiple_measurement(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(2)

        # Ensure no qubits are measured initially
        assert len(circuit.measured_qubits) == 0

        # Apply the measurement gate
        circuit.measure([0, 1])

        # Ensure both qubits are measured
        assert circuit.measured_qubits == {0, 1}

        # Define the equivalent `pytket.Circuit` instance, and
        # ensure they are equivalent
        tket_circuit = TKCircuit(2, 2)
        tket_circuit = tket_circuit.Measure(0, 0)
        tket_circuit = tket_circuit.Measure(1, 1)

        assert circuit.circuit == tket_circuit

    def test_get_statevector(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(1)

        # Create a state with non-zero real and imaginary components
        circuit.H(0)
        circuit.T(0)

        # Get the statevector of the circuit, and ensure it is correct
        statevector = circuit.get_statevector()

        assert_almost_equal(statevector, [np.sqrt(1/2), 0.5 + 0.5j], 8)

    def test_get_unitary(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(4)

        # Apply the gate
        circuit.MCX([0, 1], [2, 3])

        # Define the unitary
        unitary = circuit.get_unitary()

        # Define the equivalent `qickit.circuit.TKETCircuit` instance, and
        # ensure they are equivalent
        unitary_circuit = TKETCircuit(4)
        unitary_circuit.unitary(unitary, [0, 1, 2, 3])

        assert_almost_equal(unitary_circuit.get_unitary(), unitary, 8)

    def test_partial_get_counts(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(3)

        # Prepare the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Perform partial measurement on the first qubit and ensure the counts are correct
        circuit.measure(0)
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {"0": 512, "1": 512}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the second qubit and ensure the counts are correct
        circuit.measure(1)
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {"0": 512, "1": 512}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the third qubit and ensure the counts are correct
        circuit.measure(2)
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {"0": 1024, "1": 0}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the first and second qubits and ensure the counts are correct
        circuit.measure([0, 1])
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {'00': 512, '01': 0, '10': 0, '11': 512}) > 0.95

        circuit = circuit._remove_measurements()

        # Perform partial measurement on the first and third qubits and ensure the counts are correct
        circuit.measure([0, 2])
        counts = circuit.get_counts(1024)
        assert cosine_similarity(counts, {'00': 512, '01': 512, '10': 0, '11': 0}) > 0.95

    def test_get_counts(self) -> None:
        # Define the `qickit.circuit.TKETCircuit` instance
        circuit = TKETCircuit(2)

        # Apply the Bell state
        circuit.H(0)
        circuit.CX(0, 1)

        # Measure the circuit
        circuit.measure_all()

        # Get the counts of the circuit, and ensure it is correct
        counts = circuit.get_counts(1024)

        assert cosine_similarity(counts, {"00": 512, "01": 0, "10":0, "11": 512}) > 0.95