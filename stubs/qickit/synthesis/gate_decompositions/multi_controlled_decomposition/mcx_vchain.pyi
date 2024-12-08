from qickit.circuit import Circuit

__all__ = ["mcx_vchain_decomposition"]

def mcx_vchain_decomposition(num_control_qubits: int, output_framework: type[Circuit]) -> Circuit: ...
