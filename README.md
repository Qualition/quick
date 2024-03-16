# Qickit
`Qickit` is an agnostic gate-based circuit SDK, providing an integrated interface for using any supported quantum circuit framework seamlessly.

## Getting Started

### Prerequisites

- python 3.9+

### Installation

qickit can be installed with the command:

```
pip install qickit
```

The default installation of qickit includes numpy, qiskit, pennylane, cirq, and pytket.

## Usage

The docs/examples are a good way for understanding how qickit works. Depending on your preference, you may use the package as end-to-end, or use it in parts for low-level modifications.

## Testing

Run all tests with the command:

```
py -m pytest tests
```

Note: if you have installed in a virtual environment, remember to install pytest in the same environment using:

```
pip install pytest
```

## License

Distributed under Apache v2.0 License. See [`LICENSE`](LICENSE) for details.

## Citation

If you wish to attribute/distribute our work, please cite the accompanying paper:
```
@article{malekaninezhad2024qickit,
   title={qickit: {A} {H}igh-{L}evel {P}ython {L}ibrary for {I}ntegrating {Q}uantum {G}ate-based {F}rameworks},
   author={Amir Ali Malekani Nezhad},
   year={2024},
   journal={arXiv preprint arXiv:TBD},
}
```