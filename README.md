# Qickit
[![Tests](https://github.com/qualition/qickit/actions/workflows/tests.yml/badge.svg)](https://github.com/qualition/qickit/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/qualition/qickit/graph/badge.svg?token=IHWJZG8VJT)](https://codecov.io/gh/qualition/qickit)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e287a2eed9e24d5e9d4a3ffe911ce6a5)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![PyPI version](https://img.shields.io/pypi/v/qoin)](//pypi.org/project/qickit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![image](https://github.com/Qualition/QICKIT/assets/73689800/6878b3cd-0bd7-4b11-86db-189cb241a3f8)

NOTE: `Qickit` is still in development, and cannot be pip installed.

`Qickit` is an agnostic gate-based circuit SDK, providing an integrated interface for using any supported quantum circuit framework seamlessly.

## Getting Started

### Prerequisites

- python 3.11

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
