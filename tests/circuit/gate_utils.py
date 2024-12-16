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

__all__ = [
    "Identity_unitary_matrix",
    "X_unitary_matrix",
    "Y_unitary_matrix",
    "Z_unitary_matrix",
    "H_unitary_matrix",
    "S_unitary_matrix",
    "T_unitary_matrix",
    "Phase_unitary_matrix",
    "XPow_unitary_matrix",
    "XPow_global_shift_unitary_matrix",
    "YPow_unitary_matrix",
    "YPow_global_shift_unitary_matrix",
    "ZPow_unitary_matrix",
    "ZPow_global_shift_unitary_matrix",
    "RXX_unitary_matrix_pi_over_4_01qubits",
    "RXX_unitary_matrix_pi_over_4_10qubits",
    "RXX_unitary_matrix_1_over_4_02qubits",
    "RXX_unitary_matrix_1_over_4_20qubits",
    "RXX_unitary_matrix_pi_over_4_12qubits",
    "RYY_unitary_matrix_pi_over_4_01qubits",
    "RYY_unitary_matrix_pi_over_4_10qubits",
    "RYY_unitary_matrix_1_over_4_02qubits",
    "RYY_unitary_matrix_1_over_4_20qubits",
    "RYY_unitary_matrix_pi_over_4_12qubits",
    "RZZ_unitary_matrix_pi_over_4_01qubits",
    "RZZ_unitary_matrix_pi_over_4_10qubits",
    "RZZ_unitary_matrix_1_over_4_02qubits",
    "RZZ_unitary_matrix_1_over_4_20qubits",
    "RZZ_unitary_matrix_pi_over_4_12qubits",
    "U3_unitary_matrix",
    "SWAP_unitary_matrix_01qubits",
    "SWAP_unitary_matrix_10qubits",
    "SWAP_unitary_matrix_02qubits",
    "SWAP_unitary_matrix_20qubits",
    "SWAP_unitary_matrix_12qubits",
    "CX_unitary_matrix_01qubits",
    "CX_unitary_matrix_10qubits",
    "CX_unitary_matrix_02qubits",
    "CX_unitary_matrix_20qubits",
    "CX_unitary_matrix_12qubits",
    "CY_unitary_matrix_01qubits",
    "CY_unitary_matrix_10qubits",
    "CY_unitary_matrix_02qubits",
    "CY_unitary_matrix_20qubits",
    "CY_unitary_matrix_12qubits",
    "CZ_unitary_matrix_01qubits",
    "CZ_unitary_matrix_10qubits",
    "CZ_unitary_matrix_02qubits",
    "CZ_unitary_matrix_20qubits",
    "CZ_unitary_matrix_12qubits",
    "CH_unitary_matrix_01qubits",
    "CH_unitary_matrix_10qubits",
    "CH_unitary_matrix_02qubits",
    "CH_unitary_matrix_20qubits",
    "CH_unitary_matrix_12qubits",
    "CS_unitary_matrix_01qubits",
    "CS_unitary_matrix_10qubits",
    "CS_unitary_matrix_02qubits",
    "CS_unitary_matrix_20qubits",
    "CS_unitary_matrix_12qubits",
    "CT_unitary_matrix_01qubits",
    "CT_unitary_matrix_10qubits",
    "CT_unitary_matrix_02qubits",
    "CT_unitary_matrix_20qubits",
    "CT_unitary_matrix_12qubits",
    "CSdg_unitary_matrix_01qubits",
    "CSdg_unitary_matrix_10qubits",
    "CSdg_unitary_matrix_02qubits",
    "CSdg_unitary_matrix_20qubits",
    "CSdg_unitary_matrix_12qubits",
    "CTdg_unitary_matrix_01qubits",
    "CTdg_unitary_matrix_10qubits",
    "CTdg_unitary_matrix_02qubits",
    "CTdg_unitary_matrix_20qubits",
    "CTdg_unitary_matrix_12qubits",
    "CRX_unitary_matrix_pi_over_4_01qubits",
    "CRX_unitary_matrix_pi_over_4_10qubits",
    "CRX_unitary_matrix_1_over_4_02qubits",
    "CRX_unitary_matrix_1_over_4_20qubits",
    "CRX_unitary_matrix_pi_over_4_12qubits",
    "CRY_unitary_matrix_pi_over_4_01qubits",
    "CRY_unitary_matrix_pi_over_4_10qubits",
    "CRY_unitary_matrix_1_over_4_02qubits",
    "CRY_unitary_matrix_1_over_4_20qubits",
    "CRY_unitary_matrix_pi_over_4_12qubits",
    "CRZ_unitary_matrix_pi_over_4_01qubits",
    "CRZ_unitary_matrix_pi_over_4_10qubits",
    "CRZ_unitary_matrix_1_over_4_02qubits",
    "CRZ_unitary_matrix_1_over_4_20qubits",
    "CRZ_unitary_matrix_pi_over_4_12qubits",
    "CPhase_unitary_matrix_pi_over_4_01qubits",
    "CPhase_unitary_matrix_pi_over_4_10qubits",
    "CPhase_unitary_matrix_1_over_4_02qubits",
    "CPhase_unitary_matrix_1_over_4_20qubits",
    "CPhase_unitary_matrix_pi_over_4_12qubits",
    "CXPow_unitary_matrix_1_over_4_0_shift_01qubits",
    "CXPow_unitary_matrix_1_over_4_0_shift_10qubits",
    "CXPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits",
    "CXPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits",
    "CXPow_unitary_matrix_1_over_4_0_shift_12qubits",
    "CYPow_unitary_matrix_1_over_4_0_shift_01qubits",
    "CYPow_unitary_matrix_1_over_4_0_shift_10qubits",
    "CYPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits",
    "CYPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits",
    "CYPow_unitary_matrix_1_over_4_0_shift_12qubits",
    "CZPow_unitary_matrix_1_over_4_0_shift_01qubits",
    "CZPow_unitary_matrix_1_over_4_0_shift_10qubits",
    "CZPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits",
    "CZPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits",
    "CZPow_unitary_matrix_1_over_4_0_shift_12qubits",
    "CRXX_unitary_matrix_pi_over_4_012qubits",
    "CRXX_unitary_matrix_pi_over_4_102qubits",
    "CRXX_unitary_matrix_1_over_4_123qubits",
    "CRXX_unitary_matrix_1_over_4_213qubits",
    "CRXX_unitary_matrix_pi_over_4_023qubits",
    "CRYY_unitary_matrix_pi_over_4_012qubits",
    "CRYY_unitary_matrix_pi_over_4_102qubits",
    "CRYY_unitary_matrix_1_over_4_123qubits",
    "CRYY_unitary_matrix_1_over_4_213qubits",
    "CRYY_unitary_matrix_pi_over_4_023qubits",
    "CRZZ_unitary_matrix_pi_over_4_012qubits",
    "CRZZ_unitary_matrix_pi_over_4_102qubits",
    "CRZZ_unitary_matrix_1_over_4_123qubits",
    "CRZZ_unitary_matrix_1_over_4_213qubits",
    "CRZZ_unitary_matrix_pi_over_4_023qubits",
    "CU3_unitary_matrix_pi2_pi3_pi4_01qubits",
    "CU3_unitary_matrix_pi2_pi3_pi4_10qubits",
    "CU3_unitary_matrix_pi2_pi3_pi4_02qubits",
    "CU3_unitary_matrix_pi2_pi3_pi4_20qubits",
    "CU3_unitary_matrix_pi2_pi3_pi4_12qubits",
    "CSWAP_unitary_matrix_012qubits",
    "CSWAP_unitary_matrix_102qubits",
    "CSWAP_unitary_matrix_123qubits",
    "CSWAP_unitary_matrix_213qubits",
    "CSWAP_unitary_matrix_023qubits",
    "MCX_unitary_matrix_01_23_qubits",
    "MCX_unitary_matrix_10_23_qubits",
    "MCX_unitary_matrix_02_13_qubits",
    "MCX_unitary_matrix_20_34_qubits",
    "MCX_unitary_matrix_12_04_qubits",
    "MCX_unitary_matrix_53_01_qubits",
    "MCX_unitary_matrix_012_34_qubits",
    "MCX_unitary_matrix_01_234_qubits",
    "MCX_unitary_matrix_012_345_qubits",
    "MCX_unitary_matrix_01_2_qubits",
    "MCX_unitary_matrix_0_23_qubits",
    "MCY_unitary_matrix_01_23_qubits",
    "MCY_unitary_matrix_10_23_qubits",
    "MCY_unitary_matrix_02_13_qubits",
    "MCY_unitary_matrix_20_34_qubits",
    "MCY_unitary_matrix_12_04_qubits",
    "MCY_unitary_matrix_53_01_qubits",
    "MCY_unitary_matrix_012_34_qubits",
    "MCY_unitary_matrix_01_234_qubits",
    "MCY_unitary_matrix_012_345_qubits",
    "MCY_unitary_matrix_01_2_qubits",
    "MCY_unitary_matrix_0_23_qubits",
    "MCZ_unitary_matrix_01_23_qubits",
    "MCZ_unitary_matrix_10_23_qubits",
    "MCZ_unitary_matrix_02_13_qubits",
    "MCZ_unitary_matrix_20_34_qubits",
    "MCZ_unitary_matrix_12_04_qubits",
    "MCZ_unitary_matrix_53_01_qubits",
    "MCZ_unitary_matrix_012_34_qubits",
    "MCZ_unitary_matrix_01_234_qubits",
    "MCZ_unitary_matrix_012_345_qubits",
    "MCZ_unitary_matrix_01_2_qubits",
    "MCZ_unitary_matrix_0_23_qubits",
    "MCH_unitary_matrix_01_23_qubits",
    "MCH_unitary_matrix_10_23_qubits",
    "MCH_unitary_matrix_02_13_qubits",
    "MCH_unitary_matrix_20_34_qubits",
    "MCH_unitary_matrix_12_04_qubits",
    "MCH_unitary_matrix_53_01_qubits",
    "MCH_unitary_matrix_012_34_qubits",
    "MCH_unitary_matrix_01_234_qubits",
    "MCH_unitary_matrix_012_345_qubits",
    "MCH_unitary_matrix_01_2_qubits",
    "MCH_unitary_matrix_0_23_qubits",
    "MCS_unitary_matrix_01_23_qubits",
    "MCS_unitary_matrix_10_23_qubits",
    "MCS_unitary_matrix_02_13_qubits",
    "MCS_unitary_matrix_20_34_qubits",
    "MCS_unitary_matrix_12_04_qubits",
    "MCS_unitary_matrix_53_01_qubits",
    "MCS_unitary_matrix_012_34_qubits",
    "MCS_unitary_matrix_01_234_qubits",
    "MCS_unitary_matrix_012_345_qubits",
    "MCS_unitary_matrix_01_2_qubits",
    "MCS_unitary_matrix_0_23_qubits",
    "MCT_unitary_matrix_01_23_qubits",
    "MCT_unitary_matrix_10_23_qubits",
    "MCT_unitary_matrix_02_13_qubits",
    "MCT_unitary_matrix_20_34_qubits",
    "MCT_unitary_matrix_12_04_qubits",
    "MCT_unitary_matrix_53_01_qubits",
    "MCT_unitary_matrix_012_34_qubits",
    "MCT_unitary_matrix_01_234_qubits",
    "MCT_unitary_matrix_012_345_qubits",
    "MCT_unitary_matrix_01_2_qubits",
    "MCT_unitary_matrix_0_23_qubits",
    "MCSdg_unitary_matrix_01_23_qubits",
    "MCSdg_unitary_matrix_10_23_qubits",
    "MCSdg_unitary_matrix_02_13_qubits",
    "MCSdg_unitary_matrix_20_34_qubits",
    "MCSdg_unitary_matrix_12_04_qubits",
    "MCSdg_unitary_matrix_53_01_qubits",
    "MCSdg_unitary_matrix_012_34_qubits",
    "MCSdg_unitary_matrix_01_234_qubits",
    "MCSdg_unitary_matrix_012_345_qubits",
    "MCSdg_unitary_matrix_01_2_qubits",
    "MCSdg_unitary_matrix_0_23_qubits",
    "MCTdg_unitary_matrix_01_23_qubits",
    "MCTdg_unitary_matrix_10_23_qubits",
    "MCTdg_unitary_matrix_02_13_qubits",
    "MCTdg_unitary_matrix_20_34_qubits",
    "MCTdg_unitary_matrix_12_04_qubits",
    "MCTdg_unitary_matrix_53_01_qubits",
    "MCTdg_unitary_matrix_012_34_qubits",
    "MCTdg_unitary_matrix_01_234_qubits",
    "MCTdg_unitary_matrix_012_345_qubits",
    "MCTdg_unitary_matrix_01_2_qubits",
    "MCTdg_unitary_matrix_0_23_qubits",
    "MCRX_unitary_matrix_pi_over_4_01_23_qubits",
    "MCRX_unitary_matrix_pi_over_4_10_23_qubits",
    "MCRX_unitary_matrix_1_over_4_02_13_qubits",
    "MCRX_unitary_matrix_1_over_4_20_34_qubits",
    "MCRX_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCRX_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCRX_unitary_matrix_1_over_3_012_34_qubits",
    "MCRX_unitary_matrix_1_over_3_01_234_qubits",
    "MCRX_unitary_matrix_pi_over_4_012_345_qubits",
    "MCRX_unitary_matrix_pi_over_4_01_2_qubits",
    "MCRX_unitary_matrix_pi_over_4_0_23_qubits",
    "MCRX_unitary_matrix_0dot1_012345_6_qubits",
    "MCRX_unitary_matrix_0dot1_0123456_7_qubits",
    "MCRX_unitary_matrix_0dot1_01234567_8_qubits",
    "MCRY_unitary_matrix_pi_over_4_01_23_qubits",
    "MCRY_unitary_matrix_pi_over_4_10_23_qubits",
    "MCRY_unitary_matrix_1_over_4_02_13_qubits",
    "MCRY_unitary_matrix_1_over_4_20_34_qubits",
    "MCRY_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCRY_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCRY_unitary_matrix_1_over_3_012_34_qubits",
    "MCRY_unitary_matrix_1_over_3_01_234_qubits",
    "MCRY_unitary_matrix_pi_over_4_012_345_qubits",
    "MCRY_unitary_matrix_pi_over_4_01_2_qubits",
    "MCRY_unitary_matrix_pi_over_4_0_23_qubits",
    "MCRY_unitary_matrix_0dot1_012345_6_qubits",
    "MCRY_unitary_matrix_0dot1_0123456_7_qubits",
    "MCRY_unitary_matrix_0dot1_01234567_8_qubits",
    "MCRZ_unitary_matrix_pi_over_4_01_23_qubits",
    "MCRZ_unitary_matrix_pi_over_4_10_23_qubits",
    "MCRZ_unitary_matrix_1_over_4_02_13_qubits",
    "MCRZ_unitary_matrix_1_over_4_20_34_qubits",
    "MCRZ_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCRZ_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCRZ_unitary_matrix_1_over_3_012_34_qubits",
    "MCRZ_unitary_matrix_1_over_3_01_234_qubits",
    "MCRZ_unitary_matrix_pi_over_4_012_345_qubits",
    "MCRZ_unitary_matrix_pi_over_4_01_2_qubits",
    "MCRZ_unitary_matrix_pi_over_4_0_23_qubits",
    "MCRZ_unitary_matrix_0dot1_012345_6_qubits",
    "MCRZ_unitary_matrix_0dot1_0123456_7_qubits",
    "MCRZ_unitary_matrix_0dot1_01234567_8_qubits",
    "MCPhase_unitary_matrix_pi_over_4_01_23_qubits",
    "MCPhase_unitary_matrix_pi_over_4_10_23_qubits",
    "MCPhase_unitary_matrix_1_over_4_02_13_qubits",
    "MCPhase_unitary_matrix_1_over_4_20_34_qubits",
    "MCPhase_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCPhase_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCPhase_unitary_matrix_1_over_3_012_34_qubits",
    "MCPhase_unitary_matrix_1_over_3_01_234_qubits",
    "MCPhase_unitary_matrix_pi_over_4_012_345_qubits",
    "MCPhase_unitary_matrix_pi_over_4_01_2_qubits",
    "MCPhase_unitary_matrix_pi_over_4_0_23_qubits",
    "MCXPow_unitary_matrix_1_over_4_0_shift_01_23_qubits",
    "MCXPow_unitary_matrix_1_over_4_0_shift_10_23_qubits",
    "MCXPow_unitary_matrix_1_over_4_0_shift_02_13_qubits",
    "MCXPow_unitary_matrix_1_over_4_0_shift_20_34_qubits",
    "MCXPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits",
    "MCXPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits",
    "MCXPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits",
    "MCXPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits",
    "MCYPow_unitary_matrix_1_over_4_0_shift_01_23_qubits",
    "MCYPow_unitary_matrix_1_over_4_0_shift_10_23_qubits",
    "MCYPow_unitary_matrix_1_over_4_0_shift_02_13_qubits",
    "MCYPow_unitary_matrix_1_over_4_0_shift_20_34_qubits",
    "MCYPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits",
    "MCYPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits",
    "MCYPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits",
    "MCYPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits",
    "MCZPow_unitary_matrix_1_over_4_0_shift_01_23_qubits",
    "MCZPow_unitary_matrix_1_over_4_0_shift_10_23_qubits",
    "MCZPow_unitary_matrix_1_over_4_0_shift_02_13_qubits",
    "MCZPow_unitary_matrix_1_over_4_0_shift_20_34_qubits",
    "MCZPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits",
    "MCZPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits",
    "MCZPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits",
    "MCZPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits",
    "MCRXX_unitary_matrix_pi_over_4_01_23_qubits",
    "MCRXX_unitary_matrix_pi_over_4_10_23_qubits",
    "MCRXX_unitary_matrix_1_over_4_02_13_qubits",
    "MCRXX_unitary_matrix_1_over_4_20_34_qubits",
    "MCRXX_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCRXX_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCRXX_unitary_matrix_1_over_3_012_34_qubits",
    "MCRXX_unitary_matrix_1_over_3_0_23_qubits",
    "MCRYY_unitary_matrix_pi_over_4_01_23_qubits",
    "MCRYY_unitary_matrix_pi_over_4_10_23_qubits",
    "MCRYY_unitary_matrix_1_over_4_02_13_qubits",
    "MCRYY_unitary_matrix_1_over_4_20_34_qubits",
    "MCRYY_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCRYY_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCRYY_unitary_matrix_1_over_3_012_34_qubits",
    "MCRYY_unitary_matrix_1_over_3_0_23_qubits",
    "MCRZZ_unitary_matrix_pi_over_4_01_23_qubits",
    "MCRZZ_unitary_matrix_pi_over_4_10_23_qubits",
    "MCRZZ_unitary_matrix_1_over_4_02_13_qubits",
    "MCRZZ_unitary_matrix_1_over_4_20_34_qubits",
    "MCRZZ_unitary_matrix_negative1_over_4_12_04_qubits",
    "MCRZZ_unitary_matrix_negative1_over_4_53_01_qubits",
    "MCRZZ_unitary_matrix_1_over_3_012_34_qubits",
    "MCRZZ_unitary_matrix_1_over_3_0_23_qubits",
    "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_01_23_qubits",
    "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_10_23_qubits",
    "MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_02_13_qubits",
    "MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_20_34_qubits",
    "MCU3_unitary_matrix_negative1_over_2_negative1_over_3_negative1_over_4_12_04_qubits",
    "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_123_45_qubits",
    "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_0_12_qubits",
    "MCSWAP_unitary_matrix_01_23_qubits",
    "MCSWAP_unitary_matrix_10_23_qubits",
    "MCSWAP_unitary_matrix_02_13_qubits",
    "MCSWAP_unitary_matrix_20_34_qubits",
    "MCSWAP_unitary_matrix_123_45_qubits",
    "MCSWAP_unitary_matrix_0_23_qubits",
    "UCRX_unitary_matrix_3qubits_01control",
    "UCRX_unitary_matrix_3qubits_10control",
    "UCRX_unitary_matrix_4qubits_023control",
    "UCRX_unitary_matrix_4qubits_213control",
    "UCRY_unitary_matrix_3qubits_01control",
    "UCRY_unitary_matrix_3qubits_10control",
    "UCRY_unitary_matrix_4qubits_023control",
    "UCRY_unitary_matrix_4qubits_213control",
    "UCRZ_unitary_matrix_3qubits_01control",
    "UCRZ_unitary_matrix_3qubits_10control",
    "UCRZ_unitary_matrix_4qubits_023control",
    "UCRZ_unitary_matrix_4qubits_213control",
    "UC_unitary_matrix_no_diagonal_no_simplification_3qubits_01control_HXHX",
    "UC_unitary_matrix_no_diagonal_no_simplification_3qubits_10control_HYHY",
    "UC_unitary_matrix_no_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY",
    "UC_unitary_matrix_no_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY",
    "UC_unitary_matrix_diagonal_no_simplification_3qubits_01control_HXHX",
    "UC_unitary_matrix_diagonal_no_simplification_3qubits_10control_HYHY",
    "UC_unitary_matrix_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY",
    "UC_unitary_matrix_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY",
    "UC_unitary_matrix_no_diagonal_simplification_3qubits_01control_HXHX",
    "UC_unitary_matrix_no_diagonal_simplification_3qubits_10control_HYHY",
    "UC_unitary_matrix_no_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY",
    "UC_unitary_matrix_no_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY",
    "UC_unitary_matrix_diagonal_simplification_3qubits_01control_HXHX",
    "UC_unitary_matrix_diagonal_simplification_3qubits_10control_HYHY",
    "UC_unitary_matrix_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY",
    "UC_unitary_matrix_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY",
    "qft_no_swap_no_inverse_approx0_5qubits",
    "qft_no_swap_no_inverse_approx0_6qubits",
    "qft_no_swap_no_inverse_approx0_7qubits",
    "qft_no_swap_no_inverse_approx0_8qubits",
    "qft_no_swap_no_inverse_approx1_5qubits",
    "qft_no_swap_no_inverse_approx1_6qubits",
    "qft_no_swap_no_inverse_approx1_7qubits",
    "qft_no_swap_no_inverse_approx1_8qubits",
    "qft_no_swap_no_inverse_approx2_5qubits",
    "qft_no_swap_no_inverse_approx2_6qubits",
    "qft_no_swap_no_inverse_approx2_7qubits",
    "qft_no_swap_no_inverse_approx2_8qubits",
    "qft_no_swap_no_inverse_approx3_5qubits",
    "qft_no_swap_no_inverse_approx3_6qubits",
    "qft_no_swap_no_inverse_approx3_7qubits",
    "qft_no_swap_no_inverse_approx3_8qubits",
    "qft_swap_no_inverse_approx0_5qubits",
    "qft_swap_no_inverse_approx0_6qubits",
    "qft_swap_no_inverse_approx0_7qubits",
    "qft_swap_no_inverse_approx0_8qubits",
    "qft_swap_no_inverse_approx1_5qubits",
    "qft_swap_no_inverse_approx1_6qubits",
    "qft_swap_no_inverse_approx1_7qubits",
    "qft_swap_no_inverse_approx1_8qubits",
    "qft_swap_no_inverse_approx2_5qubits",
    "qft_swap_no_inverse_approx2_6qubits",
    "qft_swap_no_inverse_approx2_7qubits",
    "qft_swap_no_inverse_approx2_8qubits",
    "qft_swap_no_inverse_approx3_5qubits",
    "qft_swap_no_inverse_approx3_6qubits",
    "qft_swap_no_inverse_approx3_7qubits",
    "qft_swap_no_inverse_approx3_8qubits",
    "qft_no_swap_inverse_approx0_5qubits",
    "qft_no_swap_inverse_approx0_6qubits",
    "qft_no_swap_inverse_approx0_7qubits",
    "qft_no_swap_inverse_approx0_8qubits",
    "qft_no_swap_inverse_approx1_5qubits",
    "qft_no_swap_inverse_approx1_6qubits",
    "qft_no_swap_inverse_approx1_7qubits",
    "qft_no_swap_inverse_approx1_8qubits",
    "qft_no_swap_inverse_approx2_5qubits",
    "qft_no_swap_inverse_approx2_6qubits",
    "qft_no_swap_inverse_approx2_7qubits",
    "qft_no_swap_inverse_approx2_8qubits",
    "qft_no_swap_inverse_approx3_5qubits",
    "qft_no_swap_inverse_approx3_6qubits",
    "qft_no_swap_inverse_approx3_7qubits",
    "qft_no_swap_inverse_approx3_8qubits",
    "qft_swap_inverse_approx0_5qubits",
    "qft_swap_inverse_approx0_6qubits",
    "qft_swap_inverse_approx0_7qubits",
    "qft_swap_inverse_approx0_8qubits",
    "qft_swap_inverse_approx1_5qubits",
    "qft_swap_inverse_approx1_6qubits",
    "qft_swap_inverse_approx1_7qubits",
    "qft_swap_inverse_approx1_8qubits",
    "qft_swap_inverse_approx2_5qubits",
    "qft_swap_inverse_approx2_6qubits",
    "qft_swap_inverse_approx2_7qubits",
    "qft_swap_inverse_approx2_8qubits",
    "qft_swap_inverse_approx3_5qubits",
    "qft_swap_inverse_approx3_6qubits",
    "qft_swap_inverse_approx3_7qubits",
    "qft_swap_inverse_approx3_8qubits"
]

import numpy as np


# Folder prefix
prefix = "tests/circuit/gate_matrix_checkers/"

""" circuit.Identity() tester

Parameters
----------
`qubit_indices` = 0
"""
Identity_unitary_matrix = np.load(prefix + "Identity_unitary_matrix.npy")

""" circuit.X() tester

Parameters
----------
`qubit_indices` = 0
"""
X_unitary_matrix = np.load(prefix + "X_unitary_matrix.npy")

""" circuit.Y() tester

Parameters
----------
`qubit_indices` = 0
"""
Y_unitary_matrix = np.load(prefix + "Y_unitary_matrix.npy")

""" circuit.Z() tester

Parameters
----------
`qubit_indices` = 0
"""
Z_unitary_matrix = np.load(prefix + "Z_unitary_matrix.npy")

""" circuit.H() tester

Parameters
----------
`qubit_indices` = 0
"""
H_unitary_matrix = np.load(prefix + "H_unitary_matrix.npy")

""" circuit.S() tester

Parameters
----------
`qubit_indices` = 0
"""
S_unitary_matrix = np.load(prefix + "S_unitary_matrix.npy")

""" circuit.T() tester

Parameters
----------
`qubit_indices` = 0
"""
T_unitary_matrix = np.load(prefix + "T_unitary_matrix.npy")

""" circuit.XPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`qubit_indices` = 0
"""
XPow_unitary_matrix = np.load(prefix + "XPow_unitary_matrix.npy")

""" circuit.XPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`qubit_indices` = 0
"""
XPow_global_shift_unitary_matrix = np.load(prefix + "XPow_global_shift_unitary_matrix.npy")

""" circuit.YPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`qubit_indices` = 0
"""
YPow_unitary_matrix = np.load(prefix + "YPow_unitary_matrix.npy")

""" circuit.YPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`qubit_indices` = 0
"""
YPow_global_shift_unitary_matrix = np.load(prefix + "YPow_global_shift_unitary_matrix.npy")

""" circuit.ZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`qubit_indices` = 0
"""
ZPow_unitary_matrix = np.load(prefix + "ZPow_unitary_matrix.npy")

""" circuit.ZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`qubit_indices` = 0
"""
ZPow_global_shift_unitary_matrix = np.load(prefix + "ZPow_global_shift_unitary_matrix.npy")

""" circuit.RXX() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 0
`second_qubit_index` = 1
"""
RXX_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "RXX_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.RXX() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 1
`second_qubit_index` = 0
"""
RXX_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "RXX_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.RXX() tester

Parameters
----------
`angle` = 1/4
`first_qubit_index` = 0
`second_qubit_index` = 2
"""
RXX_unitary_matrix_1_over_4_02qubits = np.load(prefix + "RXX_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.RXX() tester

Parameters
----------
`angle` = 1/4
`first_qubit_index` = 2
`second_qubit_index` = 0
"""
RXX_unitary_matrix_1_over_4_20qubits = np.load(prefix + "RXX_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.RXX() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 1
`second_qubit_index` = 2
"""
RXX_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "RXX_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.RYY() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 0
`second_qubit_index` = 1
"""
RYY_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "RYY_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.RYY() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 1
`second_qubit_index` = 0
"""
RYY_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "RYY_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.RYY() tester

Parameters
----------
`angle` = 1/4
`first_qubit_index` = 0
`second_qubit_index` = 2
"""
RYY_unitary_matrix_1_over_4_02qubits = np.load(prefix + "RYY_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.RYY() tester

Parameters
----------
`angle` = 1/4
`first_qubit_index` = 2
`second_qubit_index` = 0
"""
RYY_unitary_matrix_1_over_4_20qubits = np.load(prefix + "RYY_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.RYY() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 1
`second_qubit_index` = 2
"""
RYY_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "RYY_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.RZZ() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 0
`second_qubit_index` = 1
"""
RZZ_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "RZZ_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.RZZ() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 1
`second_qubit_index` = 0
"""
RZZ_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "RZZ_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.RZZ() tester

Parameters
----------
`angle` = 1/4
`first_qubit_index` = 0
`second_qubit_index` = 2
"""
RZZ_unitary_matrix_1_over_4_02qubits = np.load(prefix + "RZZ_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.RZZ() tester

Parameters
----------
`angle` = 1/4
`first_qubit_index` = 2
`second_qubit_index` = 0
"""
RZZ_unitary_matrix_1_over_4_20qubits = np.load(prefix + "RZZ_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.RZZ() tester

Parameters
----------
`angle` = np.pi/4
`first_qubit_index` = 1
`second_qubit_index` = 2
"""
RZZ_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "RZZ_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.SWAP() tester

Parameters
----------
`first_qubit_index` = 0
`second_qubit_index` = 1
"""
SWAP_unitary_matrix_01qubits = np.load(prefix + "SWAP_unitary_matrix_01qubits.npy")

""" circuit.SWAP() tester

Parameters
----------
`first_qubit_index` = 1
`second_qubit_index` = 0
"""
SWAP_unitary_matrix_10qubits = np.load(prefix + "SWAP_unitary_matrix_10qubits.npy")

""" circuit.SWAP() tester

Parameters
----------
`first_qubit_index` = 0
`second_qubit_index` = 2
"""
SWAP_unitary_matrix_02qubits = np.load(prefix + "SWAP_unitary_matrix_02qubits.npy")

""" circuit.SWAP() tester

Parameters
----------
`first_qubit_index` = 2
`second_qubit_index` = 0
"""
SWAP_unitary_matrix_20qubits = np.load(prefix + "SWAP_unitary_matrix_20qubits.npy")

""" circuit.SWAP() tester

Parameters
----------
`first_qubit_index` = 1
`second_qubit_index` = 2
"""
SWAP_unitary_matrix_12qubits = np.load(prefix + "SWAP_unitary_matrix_12qubits.npy")

""" circuit.CX() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CX_unitary_matrix_01qubits = np.load(prefix + "CX_unitary_matrix_01qubits.npy")

""" circuit.CX() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CX_unitary_matrix_10qubits = np.load(prefix + "CX_unitary_matrix_10qubits.npy")

""" circuit.CX() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CX_unitary_matrix_02qubits = np.load(prefix + "CX_unitary_matrix_02qubits.npy")

""" circuit.CX() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CX_unitary_matrix_20qubits = np.load(prefix + "CX_unitary_matrix_20qubits.npy")

""" circuit.CX() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CX_unitary_matrix_12qubits = np.load(prefix + "CX_unitary_matrix_12qubits.npy")

""" circuit.CY() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CY_unitary_matrix_01qubits = np.load(prefix + "CY_unitary_matrix_01qubits.npy")

""" circuit.CY() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CY_unitary_matrix_10qubits = np.load(prefix + "CY_unitary_matrix_10qubits.npy")

""" circuit.CY() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CY_unitary_matrix_02qubits = np.load(prefix + "CY_unitary_matrix_02qubits.npy")

""" circuit.CY() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CY_unitary_matrix_20qubits = np.load(prefix + "CY_unitary_matrix_20qubits.npy")

""" circuit.CY() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CY_unitary_matrix_12qubits = np.load(prefix + "CY_unitary_matrix_12qubits.npy")

""" circuit.CZ() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CZ_unitary_matrix_01qubits = np.load(prefix + "CZ_unitary_matrix_01qubits.npy")

""" circuit.CZ() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CZ_unitary_matrix_10qubits = np.load(prefix + "CZ_unitary_matrix_10qubits.npy")

""" circuit.CZ() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CZ_unitary_matrix_02qubits = np.load(prefix + "CZ_unitary_matrix_02qubits.npy")

""" circuit.CZ() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CZ_unitary_matrix_20qubits = np.load(prefix + "CZ_unitary_matrix_20qubits.npy")

""" circuit.CZ() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CZ_unitary_matrix_12qubits = np.load(prefix + "CZ_unitary_matrix_12qubits.npy")

""" circuit.CH() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CH_unitary_matrix_01qubits = np.load(prefix + "CH_unitary_matrix_01qubits.npy")

""" circuit.CH() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CH_unitary_matrix_10qubits = np.load(prefix + "CH_unitary_matrix_10qubits.npy")

""" circuit.CH() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CH_unitary_matrix_02qubits = np.load(prefix + "CH_unitary_matrix_02qubits.npy")

""" circuit.CH() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CH_unitary_matrix_20qubits = np.load(prefix + "CH_unitary_matrix_20qubits.npy")

""" circuit.CH() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CH_unitary_matrix_12qubits = np.load(prefix + "CH_unitary_matrix_12qubits.npy")

""" circuit.CS() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CS_unitary_matrix_01qubits = np.load(prefix + "CS_unitary_matrix_01qubits.npy")

""" circuit.CS() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CS_unitary_matrix_10qubits = np.load(prefix + "CS_unitary_matrix_10qubits.npy")

""" circuit.CS() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CS_unitary_matrix_02qubits = np.load(prefix + "CS_unitary_matrix_02qubits.npy")

""" circuit.CS() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CS_unitary_matrix_20qubits = np.load(prefix + "CS_unitary_matrix_20qubits.npy")

""" circuit.CS() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CS_unitary_matrix_12qubits = np.load(prefix + "CS_unitary_matrix_12qubits.npy")

""" circuit.CT() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CT_unitary_matrix_01qubits = np.load(prefix + "CT_unitary_matrix_01qubits.npy")

""" circuit.CT() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CT_unitary_matrix_10qubits = np.load(prefix + "CT_unitary_matrix_10qubits.npy")

""" circuit.CT() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CT_unitary_matrix_02qubits = np.load(prefix + "CT_unitary_matrix_02qubits.npy")

""" circuit.CT() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CT_unitary_matrix_20qubits = np.load(prefix + "CT_unitary_matrix_20qubits.npy")

""" circuit.CT() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CT_unitary_matrix_12qubits = np.load(prefix + "CT_unitary_matrix_12qubits.npy")

""" circuit.CTdg() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CTdg_unitary_matrix_01qubits = np.load(prefix + "CTdg_unitary_matrix_01qubits.npy")

""" circuit.CTdg() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CTdg_unitary_matrix_10qubits = np.load(prefix + "CTdg_unitary_matrix_10qubits.npy")

""" circuit.CTdg() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CTdg_unitary_matrix_02qubits = np.load(prefix + "CTdg_unitary_matrix_02qubits.npy")

""" circuit.CTdg() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CTdg_unitary_matrix_20qubits = np.load(prefix + "CTdg_unitary_matrix_20qubits.npy")

""" circuit.CTdg() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CTdg_unitary_matrix_12qubits = np.load(prefix + "CTdg_unitary_matrix_12qubits.npy")

""" circuit.CSdg() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CSdg_unitary_matrix_01qubits = np.load(prefix + "CSdg_unitary_matrix_01qubits.npy")

""" circuit.CSdg() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CSdg_unitary_matrix_10qubits = np.load(prefix + "CSdg_unitary_matrix_10qubits.npy")

""" circuit.CSdg() tester

Parameters
----------
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CSdg_unitary_matrix_02qubits = np.load(prefix + "CSdg_unitary_matrix_02qubits.npy")

""" circuit.CSdg() tester

Parameters
----------
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CSdg_unitary_matrix_20qubits = np.load(prefix + "CSdg_unitary_matrix_20qubits.npy")

""" circuit.CSdg() tester

Parameters
----------
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CSdg_unitary_matrix_12qubits = np.load(prefix + "CSdg_unitary_matrix_12qubits.npy")

""" circuit.CRX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CRX_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "CRX_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.CRX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CRX_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "CRX_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.CRX() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CRX_unitary_matrix_1_over_4_02qubits = np.load(prefix + "CRX_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.CRX() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CRX_unitary_matrix_1_over_4_20qubits = np.load(prefix + "CRX_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.CRX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CRX_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "CRX_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.CRY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CRY_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "CRY_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.CRY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CRY_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "CRY_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.CRY() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CRY_unitary_matrix_1_over_4_02qubits = np.load(prefix + "CRY_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.CRY() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CRY_unitary_matrix_1_over_4_20qubits = np.load(prefix + "CRY_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.CRY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CRY_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "CRY_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.CRZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CRZ_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "CRZ_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.CRZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CRZ_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "CRZ_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.CRZ() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CRZ_unitary_matrix_1_over_4_02qubits = np.load(prefix + "CRZ_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.CRZ() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CRZ_unitary_matrix_1_over_4_20qubits = np.load(prefix + "CRZ_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.CRZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CRZ_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "CRZ_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.CPhase() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CPhase_unitary_matrix_pi_over_4_01qubits = np.load(prefix + "CPhase_unitary_matrix_pi_over_4_01qubits.npy")

""" circuit.CPhase() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CPhase_unitary_matrix_pi_over_4_10qubits = np.load(prefix + "CPhase_unitary_matrix_pi_over_4_10qubits.npy")

""" circuit.CPhase() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CPhase_unitary_matrix_1_over_4_02qubits = np.load(prefix + "CPhase_unitary_matrix_1_over_4_02qubits.npy")

""" circuit.CPhase() tester

Parameters
----------
`angle` = 1/4
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CPhase_unitary_matrix_1_over_4_20qubits = np.load(prefix + "CPhase_unitary_matrix_1_over_4_20qubits.npy")

""" circuit.CPhase() tester

Parameters
----------
`angle` = np.pi/4
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CPhase_unitary_matrix_pi_over_4_12qubits = np.load(prefix + "CPhase_unitary_matrix_pi_over_4_12qubits.npy")

""" circuit.CXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CXPow_unitary_matrix_1_over_4_0_shift_01qubits = np.load(prefix + "CXPow_unitary_matrix_1_over_4_0_shift_01qubits.npy")

""" circuit.CXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CXPow_unitary_matrix_1_over_4_0_shift_10qubits = np.load(prefix + "CXPow_unitary_matrix_1_over_4_0_shift_10qubits.npy")

""" circuit.CXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CXPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits = np.load(prefix + "CXPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits.npy")

""" circuit.CXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CXPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits = np.load(prefix + "CXPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits.npy")

""" circuit.CXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CXPow_unitary_matrix_negative1_over_4_0_shift_12qubits = np.load(prefix + "CXPow_unitary_matrix_negative1_over_4_0_shift_12qubits.npy")

""" circuit.CYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CYPow_unitary_matrix_1_over_4_0_shift_01qubits = np.load(prefix + "CYPow_unitary_matrix_1_over_4_0_shift_01qubits.npy")

""" circuit.CYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CYPow_unitary_matrix_1_over_4_0_shift_10qubits = np.load(prefix + "CYPow_unitary_matrix_1_over_4_0_shift_10qubits.npy")

""" circuit.CYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CYPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits = np.load(prefix + "CYPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits.npy")

""" circuit.CYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CYPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits = np.load(prefix + "CYPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits.npy")

""" circuit.CYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CYPow_unitary_matrix_negative1_over_4_0_shift_12qubits = np.load(prefix + "CYPow_unitary_matrix_negative1_over_4_0_shift_12qubits.npy")

""" circuit.CZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 0
`target_qubit_index` = 1
"""
CZPow_unitary_matrix_1_over_4_0_shift_01qubits = np.load(prefix + "CZPow_unitary_matrix_1_over_4_0_shift_01qubits.npy")

""" circuit.CZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 1
`target_qubit_index` = 0
"""
CZPow_unitary_matrix_1_over_4_0_shift_10qubits = np.load(prefix + "CZPow_unitary_matrix_1_over_4_0_shift_10qubits.npy")

""" circuit.CZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubit_index` = 0
`target_qubit_index` = 2
"""
CZPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits = np.load(prefix + "CZPow_unitary_matrix_1_over_4_1_over_3_shift_02qubits.npy")

""" circuit.CZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubit_index` = 2
`target_qubit_index` = 0
"""
CZPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits = np.load(prefix + "CZPow_unitary_matrix_1_over_4_1_over_3_shift_20qubits.npy")

""" circuit.CZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubit_index` = 1
`target_qubit_index` = 2
"""
CZPow_unitary_matrix_negative1_over_4_0_shift_12qubits = np.load(prefix + "CZPow_unitary_matrix_negative1_over_4_0_shift_12qubits.npy")

""" circuit.CRXX() tester

Parameters
----------
`angle` = np.pi/4
`control_index` = 0
`first_target_index` = 1
`second_target_index` = 2
"""
CRXX_unitary_matrix_pi_over_4_012qubits = np.load(prefix + "CRXX_unitary_matrix_pi_over_4_012qubits.npy")

""" circuit.CRXX() tester

Parameters
----------
`angle` = pi/4
`control_index` = 1
`first_target_index` = 0
`second_target_index` = 2
"""
CRXX_unitary_matrix_pi_over_4_102qubits = np.load(prefix + "CRXX_unitary_matrix_pi_over_4_102qubits.npy")

""" circuit.CRXX() tester

Parameters
----------
`angle` = 1/4
`control_index` = 1
`first_target_index` = 2
`second_target_index` = 3
"""
CRXX_unitary_matrix_1_over_4_123qubits = np.load(prefix + "CRXX_unitary_matrix_1_over_4_123qubits.npy")

""" circuit.CRXX() tester

Parameters
----------
`angle` = 1/4
`control_index` = 2
`first_target_index` = 1
`second_target_index` = 3
"""
CRXX_unitary_matrix_1_over_4_213qubits = np.load(prefix + "CRXX_unitary_matrix_1_over_4_213qubits.npy")

""" circuit.CRXX() tester

Parameters
----------
`angle` = np.pi/4
`control_index` = 0
`first_target_index` = 2
`second_target_index` = 3
"""
CRXX_unitary_matrix_pi_over_4_023qubits = np.load(prefix + "CRXX_unitary_matrix_pi_over_4_023qubits.npy")

""" circuit.CRYY() tester

Parameters
----------
`angle` = np.pi/4
`control_index` = 0
`first_target_index` = 1
`second_target_index` = 2
"""
CRYY_unitary_matrix_pi_over_4_012qubits = np.load(prefix + "CRYY_unitary_matrix_pi_over_4_012qubits.npy")

""" circuit.CRYY() tester

Parameters
----------
`angle` = pi/4
`control_index` = 1
`first_target_index` = 0
`second_target_index` = 2
"""
CRYY_unitary_matrix_pi_over_4_102qubits = np.load(prefix + "CRYY_unitary_matrix_pi_over_4_102qubits.npy")

""" circuit.CRYY() tester

Parameters
----------
`angle` = 1/4
`control_index` = 1
`first_target_index` = 2
`second_target_index` = 3
"""
CRYY_unitary_matrix_1_over_4_123qubits = np.load(prefix + "CRYY_unitary_matrix_1_over_4_123qubits.npy")

""" circuit.CRYY() tester

Parameters
----------
`angle` = 1/4
`control_index` = 2
`first_target_index` = 1
`second_target_index` = 3
"""
CRYY_unitary_matrix_1_over_4_213qubits = np.load(prefix + "CRYY_unitary_matrix_1_over_4_213qubits.npy")

""" circuit.CRYY() tester

Parameters
----------
`angle` = np.pi/4
`control_index` = 0
`first_target_index` = 2
`second_target_index` = 3
"""
CRYY_unitary_matrix_pi_over_4_023qubits = np.load(prefix + "CRYY_unitary_matrix_pi_over_4_023qubits.npy")

""" circuit.CRZZ() tester

Parameters
----------
`angle` = np.pi/4
`control_index` = 0
`first_target_index` = 1
`second_target_index` = 2
"""
CRZZ_unitary_matrix_pi_over_4_012qubits = np.load(prefix + "CRZZ_unitary_matrix_pi_over_4_012qubits.npy")

""" circuit.CRZZ() tester

Parameters
----------
`angle` = pi/4
`control_index` = 1
`first_target_index` = 0
`second_target_index` = 2
"""
CRZZ_unitary_matrix_pi_over_4_102qubits = np.load(prefix + "CRZZ_unitary_matrix_pi_over_4_102qubits.npy")

""" circuit.CRZZ() tester

Parameters
----------
`angle` = 1/4
`control_index` = 1
`first_target_index` = 2
`second_target_index` = 3
"""
CRZZ_unitary_matrix_1_over_4_123qubits = np.load(prefix + "CRZZ_unitary_matrix_1_over_4_123qubits.npy")

""" circuit.CRZZ() tester

Parameters
----------
`angle` = 1/4
`control_index` = 2
`first_target_index` = 1
`second_target_index` = 3
"""
CRZZ_unitary_matrix_1_over_4_213qubits = np.load(prefix + "CRZZ_unitary_matrix_1_over_4_213qubits.npy")

""" circuit.CRZZ() tester

Parameters
----------
`angle` = np.pi/4
`control_index` = 0
`first_target_index` = 2
`second_target_index` = 3
"""
CRZZ_unitary_matrix_pi_over_4_023qubits = np.load(prefix + "CRZZ_unitary_matrix_pi_over_4_023qubits.npy")

""" circuit.CU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4)
`control_index` = 0
`target_index` = 1
"""
CU3_unitary_matrix_pi2_pi3_pi4_01qubits = np.load(prefix + "CU3_unitary_matrix_pi2_pi3_pi4_01qubits.npy")

""" circuit.CU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4)
`control_index` = 1
`target_index` = 0
"""
CU3_unitary_matrix_pi2_pi3_pi4_10qubits = np.load(prefix + "CU3_unitary_matrix_pi2_pi3_pi4_10qubits.npy")

""" circuit.CU3() tester

Parameters
----------
`angles` = (np.pi/2, -np.pi/3, np.pi/4)
`control_index` = 0
`target_index` = 1
"""
CU3_unitary_matrix_pi2_pi3_pi4_02qubits = np.load(prefix + "CU3_unitary_matrix_pi2_pi3_pi4_02qubits.npy")

""" circuit.CU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, -np.pi/4)
`control_index` = 0
`target_index` = 1
"""
CU3_unitary_matrix_pi2_pi3_pi4_20qubits = np.load(prefix + "CU3_unitary_matrix_pi2_pi3_pi4_20qubits.npy")

""" circuit.CU3() tester

Parameters
----------
`angles` = (-np.pi/2, np.pi/3, np.pi/4)
`control_index` = 1
`target_index` = 2
"""
CU3_unitary_matrix_pi2_pi3_pi4_12qubits = np.load(prefix + "CU3_unitary_matrix_pi2_pi3_pi4_12qubits.npy")

""" circuit.CSWAP() tester

Parameters
----------
`control_index` = 0
`first_target_index` = 1
`second_target_index` = 2
"""
CSWAP_unitary_matrix_012qubits = np.load(prefix + "CSWAP_unitary_matrix_012qubits.npy")

""" circuit.CSWAP() tester

Parameters
----------
`control_index` = 1
`first_target_index` = 0
`second_target_index` = 2
"""
CSWAP_unitary_matrix_102qubits = np.load(prefix + "CSWAP_unitary_matrix_102qubits.npy")

""" circuit.CSWAP() tester

Parameters
----------
`control_index` = 1
`first_target_index` = 2
`second_target_index` = 3
"""
CSWAP_unitary_matrix_123qubits = np.load(prefix + "CSWAP_unitary_matrix_123qubits.npy")

""" circuit.CSWAP() tester

Parameters
----------
`control_index` = 2
`first_target_index` = 1
`second_target_index` = 3
"""
CSWAP_unitary_matrix_213qubits = np.load(prefix + "CSWAP_unitary_matrix_213qubits.npy")

""" circuit.CSWAP() tester

Parameters
----------
`control_index` = 0
`first_target_index` = 2
`second_target_index` = 3
"""
CSWAP_unitary_matrix_023qubits = np.load(prefix + "CSWAP_unitary_matrix_023qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCX_unitary_matrix_01_23_qubits = np.load(prefix + "MCX_unitary_matrix_01_23_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCX_unitary_matrix_10_23_qubits = np.load(prefix + "MCX_unitary_matrix_10_23_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCX_unitary_matrix_02_13_qubits = np.load(prefix + "MCX_unitary_matrix_02_13_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCX_unitary_matrix_20_34_qubits = np.load(prefix + "MCX_unitary_matrix_20_34_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCX_unitary_matrix_12_04_qubits = np.load(prefix + "MCX_unitary_matrix_12_04_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCX_unitary_matrix_53_01_qubits = np.load(prefix + "MCX_unitary_matrix_53_01_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCX_unitary_matrix_012_34_qubits = np.load(prefix + "MCX_unitary_matrix_012_34_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCX_unitary_matrix_01_234_qubits = np.load(prefix + "MCX_unitary_matrix_01_234_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCX_unitary_matrix_012_345_qubits = np.load(prefix + "MCX_unitary_matrix_012_345_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCX_unitary_matrix_01_2_qubits = np.load(prefix + "MCX_unitary_matrix_01_2_qubits.npy")

""" circuit.MCX() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCX_unitary_matrix_0_23_qubits = np.load(prefix + "MCX_unitary_matrix_0_23_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCY_unitary_matrix_01_23_qubits = np.load(prefix + "MCY_unitary_matrix_01_23_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCY_unitary_matrix_10_23_qubits = np.load(prefix + "MCY_unitary_matrix_10_23_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCY_unitary_matrix_02_13_qubits = np.load(prefix + "MCY_unitary_matrix_02_13_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCY_unitary_matrix_20_34_qubits = np.load(prefix + "MCY_unitary_matrix_20_34_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCY_unitary_matrix_12_04_qubits = np.load(prefix + "MCY_unitary_matrix_12_04_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCY_unitary_matrix_53_01_qubits = np.load(prefix + "MCY_unitary_matrix_53_01_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCY_unitary_matrix_012_34_qubits = np.load(prefix + "MCY_unitary_matrix_012_34_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCY_unitary_matrix_01_234_qubits = np.load(prefix + "MCY_unitary_matrix_01_234_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCY_unitary_matrix_012_345_qubits = np.load(prefix + "MCY_unitary_matrix_012_345_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCY_unitary_matrix_01_2_qubits = np.load(prefix + "MCY_unitary_matrix_01_2_qubits.npy")

""" circuit.MCY() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCY_unitary_matrix_0_23_qubits = np.load(prefix + "MCY_unitary_matrix_0_23_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCZ_unitary_matrix_01_23_qubits = np.load(prefix + "MCZ_unitary_matrix_01_23_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCZ_unitary_matrix_10_23_qubits = np.load(prefix + "MCZ_unitary_matrix_10_23_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCZ_unitary_matrix_02_13_qubits = np.load(prefix + "MCZ_unitary_matrix_02_13_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCZ_unitary_matrix_20_34_qubits = np.load(prefix + "MCZ_unitary_matrix_20_34_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCZ_unitary_matrix_12_04_qubits = np.load(prefix + "MCZ_unitary_matrix_12_04_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCZ_unitary_matrix_53_01_qubits = np.load(prefix + "MCZ_unitary_matrix_53_01_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCZ_unitary_matrix_012_34_qubits = np.load(prefix + "MCZ_unitary_matrix_012_34_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCZ_unitary_matrix_01_234_qubits = np.load(prefix + "MCZ_unitary_matrix_01_234_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCZ_unitary_matrix_012_345_qubits = np.load(prefix + "MCZ_unitary_matrix_012_345_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCZ_unitary_matrix_01_2_qubits = np.load(prefix + "MCZ_unitary_matrix_01_2_qubits.npy")

""" circuit.MCZ() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCZ_unitary_matrix_0_23_qubits = np.load(prefix + "MCZ_unitary_matrix_0_23_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCH_unitary_matrix_01_23_qubits = np.load(prefix + "MCH_unitary_matrix_01_23_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCH_unitary_matrix_10_23_qubits = np.load(prefix + "MCH_unitary_matrix_10_23_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCH_unitary_matrix_02_13_qubits = np.load(prefix + "MCH_unitary_matrix_02_13_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCH_unitary_matrix_20_34_qubits = np.load(prefix + "MCH_unitary_matrix_20_34_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCH_unitary_matrix_12_04_qubits = np.load(prefix + "MCH_unitary_matrix_12_04_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCH_unitary_matrix_53_01_qubits = np.load(prefix + "MCH_unitary_matrix_53_01_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCH_unitary_matrix_012_34_qubits = np.load(prefix + "MCH_unitary_matrix_012_34_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCH_unitary_matrix_01_234_qubits = np.load(prefix + "MCH_unitary_matrix_01_234_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCH_unitary_matrix_012_345_qubits = np.load(prefix + "MCH_unitary_matrix_012_345_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCH_unitary_matrix_01_2_qubits = np.load(prefix + "MCH_unitary_matrix_01_2_qubits.npy")

""" circuit.MCH() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCH_unitary_matrix_0_23_qubits = np.load(prefix + "MCH_unitary_matrix_0_23_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCS_unitary_matrix_01_23_qubits = np.load(prefix + "MCS_unitary_matrix_01_23_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCS_unitary_matrix_10_23_qubits = np.load(prefix + "MCS_unitary_matrix_10_23_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCS_unitary_matrix_02_13_qubits = np.load(prefix + "MCS_unitary_matrix_02_13_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCS_unitary_matrix_20_34_qubits = np.load(prefix + "MCS_unitary_matrix_20_34_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCS_unitary_matrix_12_04_qubits = np.load(prefix + "MCS_unitary_matrix_12_04_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCS_unitary_matrix_53_01_qubits = np.load(prefix + "MCS_unitary_matrix_53_01_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCS_unitary_matrix_012_34_qubits = np.load(prefix + "MCS_unitary_matrix_012_34_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCS_unitary_matrix_01_234_qubits = np.load(prefix + "MCS_unitary_matrix_01_234_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCS_unitary_matrix_012_345_qubits = np.load(prefix + "MCS_unitary_matrix_012_345_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCS_unitary_matrix_01_2_qubits = np.load(prefix + "MCS_unitary_matrix_01_2_qubits.npy")

""" circuit.MCS() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCS_unitary_matrix_0_23_qubits = np.load(prefix + "MCS_unitary_matrix_0_23_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCSdg_unitary_matrix_01_23_qubits = np.load(prefix + "MCSdg_unitary_matrix_01_23_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCSdg_unitary_matrix_10_23_qubits = np.load(prefix + "MCSdg_unitary_matrix_10_23_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCSdg_unitary_matrix_02_13_qubits = np.load(prefix + "MCSdg_unitary_matrix_02_13_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCSdg_unitary_matrix_20_34_qubits = np.load(prefix + "MCSdg_unitary_matrix_20_34_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCSdg_unitary_matrix_12_04_qubits = np.load(prefix + "MCSdg_unitary_matrix_12_04_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCSdg_unitary_matrix_53_01_qubits = np.load(prefix + "MCSdg_unitary_matrix_53_01_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCSdg_unitary_matrix_012_34_qubits = np.load(prefix + "MCSdg_unitary_matrix_012_34_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCSdg_unitary_matrix_01_234_qubits = np.load(prefix + "MCSdg_unitary_matrix_01_234_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCSdg_unitary_matrix_012_345_qubits = np.load(prefix + "MCSdg_unitary_matrix_012_345_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCSdg_unitary_matrix_01_2_qubits = np.load(prefix + "MCSdg_unitary_matrix_01_2_qubits.npy")

""" circuit.MCSdg() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCSdg_unitary_matrix_0_23_qubits = np.load(prefix + "MCSdg_unitary_matrix_0_23_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCT_unitary_matrix_01_23_qubits = np.load(prefix + "MCT_unitary_matrix_01_23_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCT_unitary_matrix_10_23_qubits = np.load(prefix + "MCT_unitary_matrix_10_23_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCT_unitary_matrix_02_13_qubits = np.load(prefix + "MCT_unitary_matrix_02_13_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCT_unitary_matrix_20_34_qubits = np.load(prefix + "MCT_unitary_matrix_20_34_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCT_unitary_matrix_12_04_qubits = np.load(prefix + "MCT_unitary_matrix_12_04_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCT_unitary_matrix_53_01_qubits = np.load(prefix + "MCT_unitary_matrix_53_01_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCT_unitary_matrix_012_34_qubits = np.load(prefix + "MCT_unitary_matrix_012_34_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCT_unitary_matrix_01_234_qubits = np.load(prefix + "MCT_unitary_matrix_01_234_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCT_unitary_matrix_012_345_qubits = np.load(prefix + "MCT_unitary_matrix_012_345_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCT_unitary_matrix_01_2_qubits = np.load(prefix + "MCT_unitary_matrix_01_2_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCT_unitary_matrix_0_23_qubits = np.load(prefix + "MCT_unitary_matrix_0_23_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCTdg_unitary_matrix_01_23_qubits = np.load(prefix + "MCTdg_unitary_matrix_01_23_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCTdg_unitary_matrix_10_23_qubits = np.load(prefix + "MCTdg_unitary_matrix_10_23_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCTdg_unitary_matrix_02_13_qubits = np.load(prefix + "MCTdg_unitary_matrix_02_13_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCTdg_unitary_matrix_20_34_qubits = np.load(prefix + "MCTdg_unitary_matrix_20_34_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCTdg_unitary_matrix_12_04_qubits = np.load(prefix + "MCTdg_unitary_matrix_12_04_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCTdg_unitary_matrix_53_01_qubits = np.load(prefix + "MCTdg_unitary_matrix_53_01_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCTdg_unitary_matrix_012_34_qubits = np.load(prefix + "MCTdg_unitary_matrix_012_34_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [3, 4, 5]
"""
MCTdg_unitary_matrix_01_234_qubits = np.load(prefix + "MCTdg_unitary_matrix_01_234_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCTdg_unitary_matrix_012_345_qubits = np.load(prefix + "MCTdg_unitary_matrix_012_345_qubits.npy")

""" circuit.MCTdg() tester

Parameters
----------
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCTdg_unitary_matrix_01_2_qubits = np.load(prefix + "MCTdg_unitary_matrix_01_2_qubits.npy")

""" circuit.MCT() tester

Parameters
----------
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCTdg_unitary_matrix_0_23_qubits = np.load(prefix + "MCTdg_unitary_matrix_0_23_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCRX_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCRX_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCRX_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCRX_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCRX_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCRX_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCRX_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCRX_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCRX_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCRX_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCRX_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCRX_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCRX_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCRX_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3, 4]
"""
MCRX_unitary_matrix_1_over_3_01_234_qubits = np.load(prefix + "MCRX_unitary_matrix_1_over_3_01_234_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCRX_unitary_matrix_pi_over_4_012_345_qubits = np.load(prefix + "MCRX_unitary_matrix_pi_over_4_012_345_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCRX_unitary_matrix_pi_over_4_01_2_qubits = np.load(prefix + "MCRX_unitary_matrix_pi_over_4_01_2_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCRX_unitary_matrix_pi_over_4_0_23_qubits = np.load(prefix + "MCRX_unitary_matrix_pi_over_4_0_23_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5]
`target_qubits` = [6]
"""
MCRX_unitary_matrix_0dot1_012345_6_qubits = np.load(prefix + "MCRX_unitary_matrix_0dot1_012345_6_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5, 6]
`target_qubits` = [7]
"""
MCRX_unitary_matrix_0dot1_0123456_7_qubits = np.load(prefix + "MCRX_unitary_matrix_0dot1_0123456_7_qubits.npy")

""" circuit.MCRX() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5, 6, 7]
`target_qubits` = [8]
"""
MCRX_unitary_matrix_0dot1_01234567_8_qubits = np.load(prefix + "MCRX_unitary_matrix_0dot1_01234567_8_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCRY_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCRY_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCRY_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCRY_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCRY_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCRY_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCRY_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCRY_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCRY_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCRY_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCRY_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCRY_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCRY_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCRY_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3, 4]
"""
MCRY_unitary_matrix_1_over_3_01_234_qubits = np.load(prefix + "MCRY_unitary_matrix_1_over_3_01_234_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCRY_unitary_matrix_pi_over_4_012_345_qubits = np.load(prefix + "MCRY_unitary_matrix_pi_over_4_012_345_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCRY_unitary_matrix_pi_over_4_01_2_qubits = np.load(prefix + "MCRY_unitary_matrix_pi_over_4_01_2_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCRY_unitary_matrix_pi_over_4_0_23_qubits = np.load(prefix + "MCRY_unitary_matrix_pi_over_4_0_23_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5]
`target_qubits` = [6]
"""
MCRY_unitary_matrix_0dot1_012345_6_qubits = np.load(prefix + "MCRY_unitary_matrix_0dot1_012345_6_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5, 6]
`target_qubits` = [7]
"""
MCRY_unitary_matrix_0dot1_0123456_7_qubits = np.load(prefix + "MCRY_unitary_matrix_0dot1_0123456_7_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5, 6, 7]
`target_qubits` = [8]
"""
MCRY_unitary_matrix_0dot1_01234567_8_qubits = np.load(prefix + "MCRY_unitary_matrix_0dot1_01234567_8_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCRZ_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCRZ_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCRZ_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCRZ_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCRZ_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCRZ_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCRZ_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCRZ_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCRZ_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCRZ_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCRZ_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCRZ_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCRZ_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCRZ_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3, 4]
"""
MCRZ_unitary_matrix_1_over_3_01_234_qubits = np.load(prefix + "MCRZ_unitary_matrix_1_over_3_01_234_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCRZ_unitary_matrix_pi_over_4_012_345_qubits = np.load(prefix + "MCRZ_unitary_matrix_pi_over_4_012_345_qubits.npy")

""" circuit.MCRY() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCRZ_unitary_matrix_pi_over_4_01_2_qubits = np.load(prefix + "MCRZ_unitary_matrix_pi_over_4_01_2_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCRZ_unitary_matrix_pi_over_4_0_23_qubits = np.load(prefix + "MCRZ_unitary_matrix_pi_over_4_0_23_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5]
`target_qubits` = [6]
"""
MCRZ_unitary_matrix_0dot1_012345_6_qubits = np.load(prefix + "MCRZ_unitary_matrix_0dot1_012345_6_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5, 6]
`target_qubits` = [7]
"""
MCRZ_unitary_matrix_0dot1_0123456_7_qubits = np.load(prefix + "MCRZ_unitary_matrix_0dot1_0123456_7_qubits.npy")

""" circuit.MCRZ() tester

Parameters
----------
`angle` = 0.1
`control_qubits` = [0, 1, 2, 3, 4, 5, 6, 7]
`target_qubits` = [8]
"""
MCRZ_unitary_matrix_0dot1_01234567_8_qubits = np.load(prefix + "MCRZ_unitary_matrix_0dot1_01234567_8_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCPhase_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCPhase_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCPhase_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCPhase_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCPhase_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCPhase_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCPhase_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCPhase_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCPhase_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCPhase_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`target_qubits` = [0, 1]
"""
MCPhase_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCPhase_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4]
"""
MCPhase_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCPhase_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3, 4]
"""
MCPhase_unitary_matrix_1_over_3_01_234_qubits = np.load(prefix + "MCPhase_unitary_matrix_1_over_3_01_234_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1, 2]
`target_qubits` = [3, 4, 5]
"""
MCPhase_unitary_matrix_pi_over_4_012_345_qubits = np.load(prefix + "MCPhase_unitary_matrix_pi_over_4_012_345_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCPhase_unitary_matrix_pi_over_4_01_2_qubits = np.load(prefix + "MCPhase_unitary_matrix_pi_over_4_01_2_qubits.npy")

""" circuit.MCPhase() tester

Parameters
----------
`angle` = pi/4
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCPhase_unitary_matrix_pi_over_4_0_23_qubits = np.load(prefix + "MCPhase_unitary_matrix_pi_over_4_0_23_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCXPow_unitary_matrix_1_over_4_0_shift_01_23_qubits = np.load(prefix + "MCXPow_unitary_matrix_1_over_4_0_shift_01_23_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCXPow_unitary_matrix_1_over_4_0_shift_10_23_qubits = np.load(prefix + "MCXPow_unitary_matrix_1_over_4_0_shift_10_23_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCXPow_unitary_matrix_1_over_4_0_shift_02_13_qubits = np.load(prefix + "MCXPow_unitary_matrix_1_over_4_0_shift_02_13_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCXPow_unitary_matrix_1_over_4_0_shift_20_34_qubits = np.load(prefix + "MCXPow_unitary_matrix_1_over_4_0_shift_20_34_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = -1/4
`global_shift` = 0
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCXPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits = np.load(prefix + "MCXPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCXPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits = np.load(prefix + "MCXPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCXPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits = np.load(prefix + "MCXPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits.npy")

""" circuit.MCXPow() tester

Parameters
----------
`power` = -1/4
`global_shift` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCXPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits = np.load(prefix + "MCXPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCYPow_unitary_matrix_1_over_4_0_shift_01_23_qubits = np.load(prefix + "MCYPow_unitary_matrix_1_over_4_0_shift_01_23_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCYPow_unitary_matrix_1_over_4_0_shift_10_23_qubits = np.load(prefix + "MCYPow_unitary_matrix_1_over_4_0_shift_10_23_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCYPow_unitary_matrix_1_over_4_0_shift_02_13_qubits = np.load(prefix + "MCYPow_unitary_matrix_1_over_4_0_shift_02_13_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCYPow_unitary_matrix_1_over_4_0_shift_20_34_qubits = np.load(prefix + "MCYPow_unitary_matrix_1_over_4_0_shift_20_34_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = -1/4
`global_shift` = 0
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCYPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits = np.load(prefix + "MCYPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCYPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits = np.load(prefix + "MCYPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCYPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits = np.load(prefix + "MCYPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits.npy")

""" circuit.MCYPow() tester

Parameters
----------
`power` = -1/4
`global_shift` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCYPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits = np.load(prefix + "MCYPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCZPow_unitary_matrix_1_over_4_0_shift_01_23_qubits = np.load(prefix + "MCZPow_unitary_matrix_1_over_4_0_shift_01_23_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCZPow_unitary_matrix_1_over_4_0_shift_10_23_qubits = np.load(prefix + "MCZPow_unitary_matrix_1_over_4_0_shift_10_23_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCZPow_unitary_matrix_1_over_4_0_shift_02_13_qubits = np.load(prefix + "MCZPow_unitary_matrix_1_over_4_0_shift_02_13_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 0
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCZPow_unitary_matrix_1_over_4_0_shift_20_34_qubits = np.load(prefix + "MCZPow_unitary_matrix_1_over_4_0_shift_20_34_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = -1/4
`global_shift` = 0
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCZPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits = np.load(prefix + "MCZPow_unitary_matrix_negative1_over_4_0_shift_12_04_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2]
"""
MCZPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits = np.load(prefix + "MCZPow_unitary_matrix_1_over_4_1_over_3_shift_01_2_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = 1/4
`global_shift` = 1/3
`control_qubits` = [0]
`target_qubits` = [2, 3]
"""
MCZPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits = np.load(prefix + "MCZPow_unitary_matrix_1_over_4_1_over_3_shift_0_23_qubits.npy")

""" circuit.MCZPow() tester

Parameters
----------
`power` = -1/4
`global_shift` = 1/3
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCZPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits = np.load(prefix + "MCZPow_unitary_matrix_negative1_over_4_negative1_over_3_shift_01_23_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRXX_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCRXX_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRXX_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCRXX_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`first_target_index` = 1
`second_target_index` = 3
"""
MCRXX_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCRXX_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`first_target_index` = 3
`second_target_index` = 4
"""
MCRXX_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCRXX_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`first_target_index` = 0
`second_target_index` = 4
"""
MCRXX_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCRXX_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`first_target_index` = 0
`second_target_index` = 1
"""
MCRXX_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCRXX_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`first_target_index` = 3
`second_target_index` = 4
"""
MCRXX_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCRXX_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCRXX() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRXX_unitary_matrix_1_over_3_0_23_qubits = np.load(prefix + "MCRXX_unitary_matrix_1_over_3_0_23_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRYY_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCRYY_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRYY_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCRYY_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`first_target_index` = 1
`second_target_index` = 3
"""
MCRYY_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCRYY_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`first_target_index` = 3
`second_target_index` = 4
"""
MCRYY_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCRYY_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`first_target_index` = 0
`second_target_index` = 4
"""
MCRYY_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCRYY_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`first_target_index` = 0
`second_target_index` = 1
"""
MCRYY_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCRYY_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`first_target_index` = 3
`second_target_index` = 4
"""
MCRYY_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCRYY_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCRYY() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRYY_unitary_matrix_1_over_3_0_23_qubits = np.load(prefix + "MCRYY_unitary_matrix_1_over_3_0_23_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [0, 1]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRZZ_unitary_matrix_pi_over_4_01_23_qubits = np.load(prefix + "MCRZZ_unitary_matrix_pi_over_4_01_23_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = np.pi/4
`control_qubits` = [1, 0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRZZ_unitary_matrix_pi_over_4_10_23_qubits = np.load(prefix + "MCRZZ_unitary_matrix_pi_over_4_10_23_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [0, 2]
`first_target_index` = 1
`second_target_index` = 3
"""
MCRZZ_unitary_matrix_1_over_4_02_13_qubits = np.load(prefix + "MCRZZ_unitary_matrix_1_over_4_02_13_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = 1/4
`control_qubits` = [2, 0]
`first_target_index` = 3
`second_target_index` = 4
"""
MCRZZ_unitary_matrix_1_over_4_20_34_qubits = np.load(prefix + "MCRZZ_unitary_matrix_1_over_4_20_34_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [1, 2]
`first_target_index` = 0
`second_target_index` = 4
"""
MCRZZ_unitary_matrix_negative1_over_4_12_04_qubits = np.load(prefix + "MCRZZ_unitary_matrix_negative1_over_4_12_04_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = -1/4
`control_qubits` = [5, 3]
`first_target_index` = 0
`second_target_index` = 1
"""
MCRZZ_unitary_matrix_negative1_over_4_53_01_qubits = np.load(prefix + "MCRZZ_unitary_matrix_negative1_over_4_53_01_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0, 1, 2]
`first_target_index` = 3
`second_target_index` = 4
"""
MCRZZ_unitary_matrix_1_over_3_012_34_qubits = np.load(prefix + "MCRZZ_unitary_matrix_1_over_3_012_34_qubits.npy")

""" circuit.MCRZZ() tester

Parameters
----------
`angle` = 1/3
`control_qubits` = [0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCRZZ_unitary_matrix_1_over_3_0_23_qubits = np.load(prefix + "MCRZZ_unitary_matrix_1_over_3_0_23_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4)
`control_qubits` = [0, 1]
`target_qubits` = [2, 3]
"""
MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_01_23_qubits = np.load(prefix + "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_01_23_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4)
`control_qubits` = [1, 0]
`target_qubits` = [2, 3]
"""
MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_10_23_qubits = np.load(prefix + "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_10_23_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (1/2, 1/3, 1/4)
`control_qubits` = [0, 2]
`target_qubits` = [1, 3]
"""
MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_02_13_qubits = np.load(prefix + "MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_02_13_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (1/2, 1/3, 1/4)
`control_qubits` = [2, 0]
`target_qubits` = [3, 4]
"""
MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_20_34_qubits = np.load(prefix + "MCU3_unitary_matrix_1_over_2_1_over_3_1_over_4_20_34_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (-1/2, -1/3, -1/4)
`control_qubits` = [1, 2]
`target_qubits` = [0, 4]
"""
MCU3_unitary_matrix_negative1_over_2_negative1_over_3_negative1_over_4_12_04_qubits = np.load(prefix + "MCU3_unitary_matrix_negative1_over_2_negative1_over_3_negative1_over_4_12_04_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4)
`control_qubits` = [1, 2, 3]
`target_qubits` = [4, 5]
"""
MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_123_45_qubits = np.load(prefix + "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_123_45_qubits.npy")

""" circuit.MCU3() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4)
`control_qubits` = [0]
`target_qubits` = [1, 2]
"""
MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_0_12_qubits = np.load(prefix + "MCU3_unitary_matrix_pi_over_2_pi_over_3_pi_over_4_0_12_qubits.npy")

""" circuit.MCSWAP() tester

Parameters
----------
`control_qubits` = [0, 1]
`first_target_index` = 2
`second_target_index` = 3
"""
MCSWAP_unitary_matrix_01_23_qubits = np.load(prefix + "MCSWAP_unitary_matrix_01_23_qubits.npy")

""" circuit.MCSWAP() tester

Parameters
----------
`control_qubits` = [1, 0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCSWAP_unitary_matrix_10_23_qubits = np.load(prefix + "MCSWAP_unitary_matrix_10_23_qubits.npy")

""" circuit.MCSWAP() tester

Parameters
----------
`control_qubits` = [0, 2]
`first_target_index` = 1
`second_target_index` = 3
"""
MCSWAP_unitary_matrix_02_13_qubits = np.load(prefix + "MCSWAP_unitary_matrix_02_13_qubits.npy")

""" circuit.MCSWAP() tester

Parameters
----------
`control_qubits` = [2, 0]
`first_target_index` = 3
`second_target_index` = 4
"""
MCSWAP_unitary_matrix_20_34_qubits = np.load(prefix + "MCSWAP_unitary_matrix_20_34_qubits.npy")

""" circuit.MCSWAP() tester

Parameters
----------
`control_qubits` = [1, 2, 3]
`first_target_index` = 4
`second_target_index` = 5
"""
MCSWAP_unitary_matrix_123_45_qubits = np.load(prefix + "MCSWAP_unitary_matrix_123_45_qubits.npy")

""" circuit.MCSWAP() tester

Parameters
----------
`control_qubits` = [0]
`first_target_index` = 2
`second_target_index` = 3
"""
MCSWAP_unitary_matrix_0_23_qubits = np.load(prefix + "MCSWAP_unitary_matrix_0_23_qubits.npy")

""" circuit.UCRX() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5)
`control_qubits` = [0, 1]
`target_qubits` = 2
"""
UCRX_unitary_matrix_3qubits_01control = np.load(prefix + "UCRX_unitary_matrix_3qubits_01control.npy")

""" circuit.UCRX() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5)
`control_qubits` = [1, 0]
`target_qubits` = 2
"""
UCRX_unitary_matrix_3qubits_10control = np.load(prefix + "UCRX_unitary_matrix_3qubits_10control.npy")

""" circuit.UCRX() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9)
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
"""
UCRX_unitary_matrix_4qubits_023control = np.load(prefix + "UCRX_unitary_matrix_4qubits_023control.npy")

""" circuit.UCRX() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9)
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
"""
UCRX_unitary_matrix_4qubits_213control = np.load(prefix + "UCRX_unitary_matrix_4qubits_213control.npy")

""" circuit.UCRY() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5)
`control_qubits` = [0, 1]
`target_qubits` = 2
"""
UCRY_unitary_matrix_3qubits_01control = np.load(prefix + "UCRY_unitary_matrix_3qubits_01control.npy")

""" circuit.UCRY() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5)
`control_qubits` = [1, 0]
`target_qubits` = 2
"""
UCRY_unitary_matrix_3qubits_10control = np.load(prefix + "UCRY_unitary_matrix_3qubits_10control.npy")

""" circuit.UCRY() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9)
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
"""
UCRY_unitary_matrix_4qubits_023control = np.load(prefix + "UCRY_unitary_matrix_4qubits_023control.npy")

""" circuit.UCRY() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9)
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
"""
UCRY_unitary_matrix_4qubits_213control = np.load(prefix + "UCRY_unitary_matrix_4qubits_213control.npy")

""" circuit.UCRZ() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5)
`control_qubits` = [0, 1]
`target_qubits` = 2
"""
UCRZ_unitary_matrix_3qubits_01control = np.load(prefix + "UCRZ_unitary_matrix_3qubits_01control.npy")

""" circuit.UCRZ() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5)
`control_qubits` = [1, 0]
`target_qubits` = 2
"""
UCRZ_unitary_matrix_3qubits_10control = np.load(prefix + "UCRZ_unitary_matrix_3qubits_10control.npy")

""" circuit.UCRZ() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9)
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
"""
UCRZ_unitary_matrix_4qubits_023control = np.load(prefix + "UCRZ_unitary_matrix_4qubits_023control.npy")

""" circuit.UCRZ() tester

Parameters
----------
`angles` = (np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9)
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
"""
UCRZ_unitary_matrix_4qubits_213control = np.load(prefix + "UCRZ_unitary_matrix_4qubits_213control.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix]
`control_qubits` = [0, 1]
`target_qubits` = 2
`up_to_diagonal` = False
`multiplexor_simplification` = False
"""
UC_unitary_matrix_no_diagonal_no_simplification_3qubits_01control_HXHX = np.load(prefix + "UC_unitary_matrix_no_diagonal_no_simplification_3qubits_01control_HXHX.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix]
`control_qubits` = [1, 0]
`target_qubits` = 2
`up_to_diagonal` = False
`multiplexor_simplification` = False
"""
UC_unitary_matrix_no_diagonal_no_simplification_3qubits_10control_HYHY = np.load(prefix + "UC_unitary_matrix_no_diagonal_no_simplification_3qubits_10control_HYHY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
`up_to_diagonal` = False
`multiplexor_simplification` = False
"""
UC_unitary_matrix_no_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_no_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
`up_to_diagonal` = False
`multiplexor_simplification` = False
"""
UC_unitary_matrix_no_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_no_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix]
`control_qubits` = [0, 1]
`target_qubits` = 2
`up_to_diagonal` = True
`multiplexor_simplification` = False
"""
UC_unitary_matrix_diagonal_no_simplification_3qubits_01control_HXHX = np.load(prefix + "UC_unitary_matrix_diagonal_no_simplification_3qubits_01control_HXHX.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix]
`control_qubits` = [1, 0]
`target_qubits` = 2
`up_to_diagonal` = True
`multiplexor_simplification` = False
"""
UC_unitary_matrix_diagonal_no_simplification_3qubits_10control_HYHY = np.load(prefix + "UC_unitary_matrix_diagonal_no_simplification_3qubits_10control_HYHY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
`up_to_diagonal` = True
`multiplexor_simplification` = False
"""
UC_unitary_matrix_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
`up_to_diagonal` = True
`multiplexor_simplification` = False
"""
UC_unitary_matrix_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix]
`control_qubits` = [0, 1]
`target_qubits` = 2
`up_to_diagonal` = False
`multiplexor_simplification` = True
"""
UC_unitary_matrix_no_diagonal_simplification_3qubits_01control_HXHX = np.load(prefix + "UC_unitary_matrix_no_diagonal_simplification_3qubits_01control_HXHX.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix]
`control_qubits` = [1, 0]
`target_qubits` = 2
`up_to_diagonal` = False
`multiplexor_simplification` = True
"""
UC_unitary_matrix_no_diagonal_simplification_3qubits_10control_HYHY = np.load(prefix + "UC_unitary_matrix_no_diagonal_simplification_3qubits_10control_HYHY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
`up_to_diagonal` = False
`multiplexor_simplification` = True
"""
UC_unitary_matrix_no_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_no_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
`up_to_diagonal` = False
`multiplexor_simplification` = True
"""
UC_unitary_matrix_no_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_no_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix]
`control_qubits` = [0, 1]
`target_qubits` = 2
`up_to_diagonal` = True
`multiplexor_simplification` = True
"""
UC_unitary_matrix_diagonal_simplification_3qubits_01control_HXHX = np.load(prefix + "UC_unitary_matrix_diagonal_simplification_3qubits_01control_HXHX.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix]
`control_qubits` = [1, 0]
`target_qubits` = 2
`up_to_diagonal` = True
`multiplexor_simplification` = True
"""
UC_unitary_matrix_diagonal_simplification_3qubits_10control_HYHY = np.load(prefix + "UC_unitary_matrix_diagonal_simplification_3qubits_10control_HYHY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [0, 2, 3]
`target_qubits` = 1
`up_to_diagonal` = True
`multiplexor_simplification` = True
"""
UC_unitary_matrix_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY.npy")

""" circuit.UC() tester

Parameters
----------
`gates` = [RX(np.pi/2).matrix, RY(np.pi/3).matrix, RX(np.pi/4).matrix, RY(np.pi/5).matrix, RX(np.pi/6).matrix, RY(np.pi/7).matrix, RX(np.pi/8).matrix, RY(np.pi/9).matrix]
`control_qubits` = [2, 1, 3]
`target_qubits` = 0
`up_to_diagonal` = True
`multiplexor_simplification` = True
"""
UC_unitary_matrix_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY = np.load(prefix + "UC_unitary_matrix_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY.npy")

# QFT testers for (5, 6, 7, 8) qubits
# QFT(no swap, no inverse, approximation_degree = 0)
qft_no_swap_no_inverse_approx0_5qubits = np.load(prefix + "qft_no_swap_no_inverse_approx0_5qubits.npy")
qft_no_swap_no_inverse_approx0_6qubits = np.load(prefix + "qft_no_swap_no_inverse_approx0_6qubits.npy")
qft_no_swap_no_inverse_approx0_7qubits = np.load(prefix + "qft_no_swap_no_inverse_approx0_7qubits.npy")
qft_no_swap_no_inverse_approx0_8qubits = np.load(prefix + "qft_no_swap_no_inverse_approx0_8qubits.npy")

# QFT(no swap, no inverse, approximation_degree = 1)
qft_no_swap_no_inverse_approx1_5qubits = np.load(prefix + "qft_no_swap_no_inverse_approx1_5qubits.npy")
qft_no_swap_no_inverse_approx1_6qubits = np.load(prefix + "qft_no_swap_no_inverse_approx1_6qubits.npy")
qft_no_swap_no_inverse_approx1_7qubits = np.load(prefix + "qft_no_swap_no_inverse_approx1_7qubits.npy")
qft_no_swap_no_inverse_approx1_8qubits = np.load(prefix + "qft_no_swap_no_inverse_approx1_8qubits.npy")

# QFT(no swap, no inverse, approximation_degree = 2)
qft_no_swap_no_inverse_approx2_5qubits = np.load(prefix + "qft_no_swap_no_inverse_approx2_5qubits.npy")
qft_no_swap_no_inverse_approx2_6qubits = np.load(prefix + "qft_no_swap_no_inverse_approx2_6qubits.npy")
qft_no_swap_no_inverse_approx2_7qubits = np.load(prefix + "qft_no_swap_no_inverse_approx2_7qubits.npy")
qft_no_swap_no_inverse_approx2_8qubits = np.load(prefix + "qft_no_swap_no_inverse_approx2_8qubits.npy")

# QFT(no swap, no inverse, approximation_degree = 3)
qft_no_swap_no_inverse_approx3_5qubits = np.load(prefix + "qft_no_swap_no_inverse_approx3_5qubits.npy")
qft_no_swap_no_inverse_approx3_6qubits = np.load(prefix + "qft_no_swap_no_inverse_approx3_6qubits.npy")
qft_no_swap_no_inverse_approx3_7qubits = np.load(prefix + "qft_no_swap_no_inverse_approx3_7qubits.npy")
qft_no_swap_no_inverse_approx3_8qubits = np.load(prefix + "qft_no_swap_no_inverse_approx3_8qubits.npy")

# QFT(swap, no inverse, approximation_degree = 0)
qft_swap_no_inverse_approx0_5qubits = np.load(prefix + "qft_swap_no_inverse_approx0_5qubits.npy")
qft_swap_no_inverse_approx0_6qubits = np.load(prefix + "qft_swap_no_inverse_approx0_6qubits.npy")
qft_swap_no_inverse_approx0_7qubits = np.load(prefix + "qft_swap_no_inverse_approx0_7qubits.npy")
qft_swap_no_inverse_approx0_8qubits = np.load(prefix + "qft_swap_no_inverse_approx0_8qubits.npy")

# QFT(swap, no inverse, approximation_degree = 1)
qft_swap_no_inverse_approx1_5qubits = np.load(prefix + "qft_swap_no_inverse_approx1_5qubits.npy")
qft_swap_no_inverse_approx1_6qubits = np.load(prefix + "qft_swap_no_inverse_approx1_6qubits.npy")
qft_swap_no_inverse_approx1_7qubits = np.load(prefix + "qft_swap_no_inverse_approx1_7qubits.npy")
qft_swap_no_inverse_approx1_8qubits = np.load(prefix + "qft_swap_no_inverse_approx1_8qubits.npy")

# QFT(swap, no inverse, approximation_degree = 2)
qft_swap_no_inverse_approx2_5qubits = np.load(prefix + "qft_swap_no_inverse_approx2_5qubits.npy")
qft_swap_no_inverse_approx2_6qubits = np.load(prefix + "qft_swap_no_inverse_approx2_6qubits.npy")
qft_swap_no_inverse_approx2_7qubits = np.load(prefix + "qft_swap_no_inverse_approx2_7qubits.npy")
qft_swap_no_inverse_approx2_8qubits = np.load(prefix + "qft_swap_no_inverse_approx2_8qubits.npy")

# QFT(swap, no inverse, approximation_degree = 3)
qft_swap_no_inverse_approx3_5qubits = np.load(prefix + "qft_swap_no_inverse_approx3_5qubits.npy")
qft_swap_no_inverse_approx3_6qubits = np.load(prefix + "qft_swap_no_inverse_approx3_6qubits.npy")
qft_swap_no_inverse_approx3_7qubits = np.load(prefix + "qft_swap_no_inverse_approx3_7qubits.npy")
qft_swap_no_inverse_approx3_8qubits = np.load(prefix + "qft_swap_no_inverse_approx3_8qubits.npy")

# QFT(no swap, inverse, approximation_degree = 0)
qft_no_swap_inverse_approx0_5qubits = np.load(prefix + "qft_no_swap_inverse_approx0_5qubits.npy")
qft_no_swap_inverse_approx0_6qubits = np.load(prefix + "qft_no_swap_inverse_approx0_6qubits.npy")
qft_no_swap_inverse_approx0_7qubits = np.load(prefix + "qft_no_swap_inverse_approx0_7qubits.npy")
qft_no_swap_inverse_approx0_8qubits = np.load(prefix + "qft_no_swap_inverse_approx0_8qubits.npy")

# QFT(no swap, inverse, approximation_degree = 1)
qft_no_swap_inverse_approx1_5qubits = np.load(prefix + "qft_no_swap_inverse_approx1_5qubits.npy")
qft_no_swap_inverse_approx1_6qubits = np.load(prefix + "qft_no_swap_inverse_approx1_6qubits.npy")
qft_no_swap_inverse_approx1_7qubits = np.load(prefix + "qft_no_swap_inverse_approx1_7qubits.npy")
qft_no_swap_inverse_approx1_8qubits = np.load(prefix + "qft_no_swap_inverse_approx1_8qubits.npy")

# QFT(no swap, inverse, approximation_degree = 2)
qft_no_swap_inverse_approx2_5qubits = np.load(prefix + "qft_no_swap_inverse_approx2_5qubits.npy")
qft_no_swap_inverse_approx2_6qubits = np.load(prefix + "qft_no_swap_inverse_approx2_6qubits.npy")
qft_no_swap_inverse_approx2_7qubits = np.load(prefix + "qft_no_swap_inverse_approx2_7qubits.npy")
qft_no_swap_inverse_approx2_8qubits = np.load(prefix + "qft_no_swap_inverse_approx2_8qubits.npy")

# QFT(no swap, inverse, approximation_degree = 3)
qft_no_swap_inverse_approx3_5qubits = np.load(prefix + "qft_no_swap_inverse_approx3_5qubits.npy")
qft_no_swap_inverse_approx3_6qubits = np.load(prefix + "qft_no_swap_inverse_approx3_6qubits.npy")
qft_no_swap_inverse_approx3_7qubits = np.load(prefix + "qft_no_swap_inverse_approx3_7qubits.npy")
qft_no_swap_inverse_approx3_8qubits = np.load(prefix + "qft_no_swap_inverse_approx3_8qubits.npy")

# QFT(swap, inverse, approximation_degree = 0)
qft_swap_inverse_approx0_5qubits = np.load(prefix + "qft_swap_inverse_approx0_5qubits.npy")
qft_swap_inverse_approx0_6qubits = np.load(prefix + "qft_swap_inverse_approx0_6qubits.npy")
qft_swap_inverse_approx0_7qubits = np.load(prefix + "qft_swap_inverse_approx0_7qubits.npy")
qft_swap_inverse_approx0_8qubits = np.load(prefix + "qft_swap_inverse_approx0_8qubits.npy")

# QFT(swap, inverse, approximation_degree = 1)
qft_swap_inverse_approx1_5qubits = np.load(prefix + "qft_swap_inverse_approx1_5qubits.npy")
qft_swap_inverse_approx1_6qubits = np.load(prefix + "qft_swap_inverse_approx1_6qubits.npy")
qft_swap_inverse_approx1_7qubits = np.load(prefix + "qft_swap_inverse_approx1_7qubits.npy")
qft_swap_inverse_approx1_8qubits = np.load(prefix + "qft_swap_inverse_approx1_8qubits.npy")

# QFT(swap, inverse, approximation_degree = 2)
qft_swap_inverse_approx2_5qubits = np.load(prefix + "qft_swap_inverse_approx2_5qubits.npy")
qft_swap_inverse_approx2_6qubits = np.load(prefix + "qft_swap_inverse_approx2_6qubits.npy")
qft_swap_inverse_approx2_7qubits = np.load(prefix + "qft_swap_inverse_approx2_7qubits.npy")
qft_swap_inverse_approx2_8qubits = np.load(prefix + "qft_swap_inverse_approx2_8qubits.npy")

# QFT(swap, inverse, approximation_degree = 3)
qft_swap_inverse_approx3_5qubits = np.load(prefix + "qft_swap_inverse_approx3_5qubits.npy")
qft_swap_inverse_approx3_6qubits = np.load(prefix + "qft_swap_inverse_approx3_6qubits.npy")
qft_swap_inverse_approx3_7qubits = np.load(prefix + "qft_swap_inverse_approx3_7qubits.npy")
qft_swap_inverse_approx3_8qubits = np.load(prefix + "qft_swap_inverse_approx3_8qubits.npy")