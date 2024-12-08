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

""" Wrapper class for using the `genQC` diffusion model in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["Diffusion"]

from genQC.pipeline.diffusion_pipeline import DiffusionPipeline # type: ignore
from genQC.inference.infer_compilation import generate_comp_tensors, convert_tensors_to_circuits # type: ignore
import genQC.util as util # type: ignore
import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Operator as QiskitOperator # type: ignore
import torch
from typing import Sequence, SupportsIndex, TYPE_CHECKING

import qickit
if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives.operator import Operator
from qickit.synthesis.unitarypreparation import UnitaryPreparation

# Set the random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


class Diffusion(UnitaryPreparation):
    """ `qickit.synthesis.unitarypreparation.Diffusion` is the class for performing unitary
    compilation using diffusion models (DMs).
    ref: https://arxiv.org/abs/2311.02041

    Notes
    -----
    `genQC` is a quantum compilation library that uses diffusion models to approximately
    compile quantum circuits. The default pre-trained model used in this class is for
    3-qubit unitaries which can be theoretically prepared using ['h', 'cx', 'z', 'ccx',
    'swap'] gate set within 12 max gates.

    For more information on `genQC`:
    - Documentation:
    https://genqc.readthedocs.io/en/latest/
    - Source code:
    https://github.com/FlorianFuerrutter/genQC
    - Publication:
    https://arxiv.org/abs/2311.02041

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `model` : str, optional, default="Floki00/qc_unitary_3qubit"
        The pre-trained model to use.
    `prompt` : str, optional, default="Compile using: ['h', 'cx', 'z', 'ccx', 'swap']"
        The prompt to use for the compilation.
    `max_num_gates` : int, optional, default=12
        The maximum number of gates to use in the compilation.
    `num_samples` : int, optional, default=128
        The number of samples to use in the compilation.
    `min_fidelity` : float, optional, default=0.9
        The minimum fidelity to accept the solution.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `model` : str
        The pre-trained model to use.
    `prompt` : str
        The prompt to use for the compilation.
    `max_num_gates` : int
        The maximum number of gates to use in the compilation.
    `num_samples` : int
        The number of samples to use in the compilation.
    `pipeline` : genQC.pipeline.diffusion_pipeline.DiffusionPipeline
        The pre-trained model pipeline.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `qickit.circuit.Circuit`.
    ValueError
        - If the minimum fidelity is not in the range [0, 1].
    """
    def __init__(
            self,
            output_framework: type[Circuit],
            model: str="Floki00/qc_unitary_3qubit",
            prompt: str="Compile using: ['h', 'cx', 'z', 'ccx', 'swap']",
            max_num_gates: int=12,
            num_samples: int=128,
            min_fidelity: float=0.99
        ) -> None:

        super().__init__(output_framework)
        self.model = model
        self.prompt = prompt
        self.max_num_gates = max_num_gates
        self.num_samples = num_samples

        if not (min_fidelity >= 0 and min_fidelity <= 1):
            raise ValueError("The minimum fidelity should be in the range [0, 1].")
        self.min_fidelity = min_fidelity

        # Determine the device to use (CPU or GPU)
        device = util.infer_torch_device()

        # Clean the memory
        util.MemoryCleaner.purge_mem()

        # Load the pre-trained model
        self.pipeline = DiffusionPipeline.from_pretrained(repo_id=model, device=device)

    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:
        """ Apply the unitary to the circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit to apply the unitary to.
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The unitary to apply.
        `qubit_indices` : int | Sequence[int]
            The qubit indices to apply the unitary to.

        Returns
        -------
        circuit : `qickit.circuit.Circuit`
            The quantum circuit with the unitary applied.

        Raises
        ------
        TypeError
            - If the unitary is not a numpy array or an Operator object.
            - If the qubit indices are not integers or a sequence of integers.
        ValueError
            - If the unitary is not a 3-qubit unitary.
            - No solution found with fidelity > 0.9.
            - If the number of qubit indices is not equal to the number of qubits
            in the unitary operator.
        IndexError
            - If the qubit indices are out of range.
        """
        if not isinstance(unitary, (np.ndarray, Operator)):
            try:
                unitary = np.array(unitary).astype(complex)
            except (ValueError, TypeError):
                raise TypeError(f"The operator must be a numpy array or an Operator object. Received {type(unitary)} instead.")

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        if not all(isinstance(qubit_index, SupportsIndex) for qubit_index in qubit_indices):
            raise TypeError("All qubit indices must be integers.")

        if not len(qubit_indices) == unitary.num_qubits:
            raise ValueError("The number of qubit indices must match the number of qubits in the unitary.")

        if self.model == "Floki00/qc_unitary_3qubit" and unitary.num_qubits != 3:
            raise ValueError("The default pre-trained model is for 3-qubit unitaries only.")

        num_qubits = unitary.num_qubits

        # As the neural network works only with real numbers, we first separate
        # the two components and create a 2 dimensional tensor for the magnitude
        # of each component
        U_r, U_i = torch.Tensor(np.real(unitary.data)), torch.Tensor(np.imag(unitary.data))
        U_tensor = torch.stack([U_r, U_i], dim=0)

        # Now we generate a tensor representation of the desired quantum circuit using the DM based on the prompt and U
        # This is also known as inference
        out_tensors = generate_comp_tensors(
            pipeline=self.pipeline,
            prompt=self.prompt,
            U=U_tensor,
            samples=self.num_samples,
            num_of_qubits=num_qubits,
            system_size=num_qubits,
            max_gates=self.max_num_gates,
            g=10
        )

        # Find the best solution in terms of the number of gates and fidelity
        qc_list, _ = convert_tensors_to_circuits(out_tensor=out_tensors, gate_pool=self.pipeline.gate_pool)

        # Find the best solution in terms of the number of gates and fidelity
        depths = []
        solution_circuits = []

        for qc in qc_list:
            qc_unitary = QiskitOperator(qc).data

            fidelity = np.abs(
                np.dot(
                    np.conj(qc_unitary.flatten()), # type: ignore
                    unitary.data.flatten()
                )
            )/2**num_qubits

            if fidelity > self.min_fidelity:
                depths.append(qc.depth())
                solution_circuits.append(qc)

        if len(depths) == 0:
            raise ValueError(f"No solution found with fidelity > {self.min_fidelity}.")

        # Find the shortest circuit with fidelity > `self.min_fidelity`
        best_qc = solution_circuits[depths.index(min(depths))]

        diffusion_circuit = qickit.circuit.Circuit.from_qiskit(best_qc, self.output_framework)

        circuit.add(diffusion_circuit, qubit_indices)

        return circuit