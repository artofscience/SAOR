# SAO

SAO framework

## Installation

To install the package for development, first clone the repository.
Then, from the project's root run `make`. This will setup a virtual
environment inside the project at `./venv`. After activating this
environment, `source ./venv/bin/active`, the `sao` package and its
dependencies are available.

The tests for `sao` are evaluated with `pytest`:

```
# minimal reporting on the test status
pytest tests/

# explicit coverage report with missing lines highlighted
pytest tests/ --cov sao --cov-report term-missing
```
