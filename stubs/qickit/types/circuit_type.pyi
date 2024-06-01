import qiskit # type: ignore
import cirq # type: ignore
import pytket # type: ignore
import pennylane as qml # type: ignore
from qickit.types.collection import Collection

__all__ = ["Circuit_Type"]

Circuit_Type: qiskit.QuantumCircuit | cirq.Circuit | pytket.Circuit | qml.QNode | Collection
