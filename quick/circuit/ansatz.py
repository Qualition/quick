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

""" Ansatz for variational quantum circuits.
"""

from __future__ import annotations

__all__ = ["Ansatz"]

from quick.circuit import Circuit

# Type hint for nested lists of floats
Params = list[list[float] | float] | list[float]


class Ansatz:
    """ `quick.circuit.Ansatz` class for parameterized quantum circuits
    which can be used as variational ansatz for quantum machine learning
    models.

    Notes
    -----
    The `Ansatz` class is a wrapper around the `quick.circuit.Circuit` class
    which provides a more user-friendly interface for parameterized quantum
    circuits, and means for variationally updating them.

    Use-cases of variational quantum circuits include:
    - Supervised QML
    - Approximate Quantum Compilation (AQC)
    - Quantum Approximate Optimization Algorithm (QAOA)
    - Variational Quantum Eigensolver (VQE)

    This class is meant to provide a simple interface for specifically updating
    the rotation angles of a parameterized quantum circuit. Users can use the class
    in the following manner:

    ```python
    from quick.circuit import Ansatz, QiskitCircuit
    from quick.circuit.circuit_utils import reshape, flatten
    from quick.random import generate_random_state
    from scipy.optimize import minimize

    # Create a parameterized quantum circuit with
    # random state initialization
    circuit = QiskitCircuit(2)
    circuit.initialize(generate_random_state(2), [0, 1])

    # Define a target state
    target_state = generate_random_state(2)

    # Create an ansatz object
    ansatz = Ansatz(circuit)

    # Define the cost function
    def cost_function(thetas, shape):
        ansatz.thetas = reshape(thetas, shape)
        return 1 - target_state.conj().T @ ansatz.ansatz.state

    initial_thetas, shape = flatten(ansatz.thetas)

    # Optimize the ansatz circuit
    result = minimize(cost_function, initial_thetas, args=(shape), method="BFGS")
    ```

    This example demonstrates how one can use the Ansatz class to perform approximate
    state preparation by variationally updating the circuit where the cost function is
    simply the fidelity between the target state and the state prepared by the ansatz.

    For simplicity, we do not bother with defining specific optimization interfaces so
    users can use whatever means of optimization they prefer as long as they update the
    parameters of the ansatz circuit per iteration. Our recommendation is to use the
    `scipy.optimize.minimize` function which provides a minimalistic and elegant interface
    to access a variety of optimization algorithms. It should however be noted that if
    the use-case requires more sophisticated optimization techniques depending on the
    cost landscape, users are urged to implement their own optimization routines and/or
    use more advanced optimization libraries.

    Lastly, `flatten` and `reshape` functions are used to convert the parameters of the
    ansatz circuit to a 1D array and vice versa. This is necessary because the optimization
    routine expects a 1D array of parameters to optimize over, while the ansatz circuit
    requires the original shape to update its definition. Feel free to use these functions
    or implement your own as needed.

    You may also make a PR to exclude the need for `flatten` and `reshape` by providing a
    more elegant way of handling the parameter updates.

    Parameters
    ----------
    `ansatz` : quick.circuit.Circuit
        The parameterized quantum circuit to be used as the ansatz.
    `ignore_global_phase` : bool, optional, default=True
        Whether to ignore the global phase when setting the parameters
        of the ansatz.

    Attributes
    ----------
    `ansatz` : quick.circuit.Circuit
        The parameterized quantum circuit to be used as the ansatz.
    `thetas` : numpy.ndarray
        The parameters of the ansatz circuit.
    `num_params` : int
        The number of parameters in the ansatz circuit.
    `num_parameterized_gates` : int
        The number of parameterized gates in the ansatz circuit.

    Raises
    ------
    TypeError
        - If the `ansatz` is not an instance of the `quick.circuit.Circuit`.

    Usage
    -----
    >>> from quick.circuit import QiskitCircuit
    >>> circuit = QiskitCircuit(2)
    >>> circuit.H(0)
    >>> circuit.CX(0, 1)
    >>> ansatz = Ansatz(circuit)
    """
    def __init__(
            self,
            ansatz: Circuit,
            ignore_global_phase: bool = True
        ) -> None:
        """ Initialize a `quick.circuit.Ansatz` instance.
        """
        if not isinstance(ansatz, Circuit):
            raise TypeError(
                "The `ansatz` must be an instance of the `quick.circuit.Circuit`. "
                f"Received {type(ansatz)} instead."
            )

        self.ansatz = ansatz
        self.ignore_global_phase = ignore_global_phase

        if not self.is_parameterized:
            raise ValueError("The `ansatz` must contain parameterized gates.")

    @property
    def thetas(self) -> Params:
        """ The parameters of the ansatz circuit.

        Returns
        -------
        Params
            The parameters of the ansatz circuit.

        Usage
        -----
        >>> ansatz.thetas
        """
        thetas: Params = []

        for gate in self.ansatz.circuit_log:
            if "angles" in gate:
                thetas.append(gate["angles"])
            elif "angle" in gate:
                if gate["gate"] == "GlobalPhase" and self.ignore_global_phase:
                    continue
                thetas.append(gate["angle"])

        return thetas

    @thetas.setter
    def thetas(
            self,
            thetas: Params
        ) -> None:
        """ Set the parameters of the ansatz circuit.

        Parameters
        ----------
        `thetas` : Params
            The parameters to set for the ansatz circuit.

        Usage
        -----
        >>> # Set the parameters of the ansatz circuit
        ... # with one U3 gate
        >>> ansatz.thetas = np.array([0.1, 0.2, 0.3])
        """
        for gate in self.ansatz.circuit_log:
            if "angles" in gate:
                gate["angles"] = thetas.pop(0)
            elif "angle" in gate:
                if gate["gate"] == "GlobalPhase" and self.ignore_global_phase:
                    continue
                gate["angle"] = thetas.pop(0)

        self.ansatz.update()

    @property
    def num_params(self) -> int:
        """ The number of parameters in the ansatz circuit.

        Returns
        -------
        int
            The number of parameters in the ansatz circuit.

        Usage
        -----
        >>> ansatz.num_params
        """
        flattened_thetas: list[float] = []

        for theta in self.thetas:
            if isinstance(theta, list):
                flattened_thetas.extend(theta)
            else:
                flattened_thetas.append(theta)

        return len(flattened_thetas)

    @property
    def num_parameterized_gates(self) -> int:
        """ The number of parameterized gates in the ansatz circuit.

        Returns
        -------
        int
            The number of parameterized gates in the ansatz circuit.

        Usage
        -----
        >>> ansatz.num_parameterized_gates
        """
        return len(self.thetas)

    @property
    def is_parameterized(self) -> bool:
        """ Check if the ansatz circuit contains parameterized gates.

        Returns
        -------
        bool
            True if the ansatz circuit contains parameterized gates,
            False otherwise.

        Usage
        -----
        >>> ansatz.is_parameterized
        """
        return len(self.thetas) > 0