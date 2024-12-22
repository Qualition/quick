# Getting Started with Developing Qickit

This document contains guidelines for contributing to the code in this
repository. This document is relevant primarily for contributions to the `qickit`
package, such as adding support for gate-based circuit frameworks, simulator backends,
and efficient state and unitary preparation schema. If you would like to contribute
applications and examples that use the `qickit` platform, please follow the instructions
for [installing qickit][official_install] instead.

[official_install]: https://qualition.github.io/qickit/latest/using/quick_start.html#install-qickit

## Quick start guide

Before getting started with development, please create a fork of this repository
if you haven't done so already and make sure to check out the latest version on
the `main` branch. After following the instruction for [setting up your
development environment](./Dev_Setup.md), you should be able to confirm that
you can run the tests and examples using your local clone. Upon making a change,
you should make a pull request with a detailed summary of what the PR proposes.
The PR will be reviewed ASAP by the team, and if accepted it will be merged
with the [`main`](https://github.com/Qualition/QICKIT/tree/main) branch.

## Code style

With regards to code format and style, the python code should follow [this guide](python_style)
and the docstring style should follow the [numpy documentation](numpy_style) style. `qickit` may
use an internal style in certain situations, and you should use them to maintain consistensy.

[python_style]: https://google.github.io/styleguide/pyguide.html
[numpy_style]: https://numpydoc.readthedocs.io/en/latest/format.html

## Testing

`qickit` tests are categorized as unit tests and integration tests on the library code.
All code added should have an accompanying test added to the appropriate spot in the
`tests` folder.