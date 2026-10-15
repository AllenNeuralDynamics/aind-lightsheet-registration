# Contributing to AIND light-sheet registration

This document is for people changing aind-lightsheet-registration: setting up, running the checks and tests,
and the conventions a change follows. Using the package is documented in [README.md](README.md).

## Setup

```bash
uv sync
```

`uv sync` creates `.venv` holding the package and the dev group. `uv` manages the environment and
the lockfile, `uv.lock`: run commands with `uv run`, add a dependency with `uv add <package>` and a
development tool with `uv add --group dev <package>`, and never install with bare `pip`. The
jupyter group adds a kernel (`uv sync --group jupyter`).

## Checks

```bash
./scripts/run_linters_and_checks.sh -c
```

Without `-c` the script only formats. With it, the script also runs ruff's linter, mypy, interrogate,
codespell and pytest with coverage, passing any arguments after `--` to pytest. It runs in the
existing `.venv` and does not sync it.

The same checks, one at a time:

```bash
uv run --frozen ruff format
uv run --frozen ruff check
uv run --frozen mypy
uv run --frozen interrogate -v src
uv run --frozen codespell --check-filenames
uv run --frozen pytest --cov aind_lightsheet_registration
```

## Tests

```bash
uv run pytest                          # the whole suite
uv run pytest tests/test_example.py    # one file
uv run pytest -k "version"             # tests whose names match
```

## Conventions

- Source lives in `src/aind_lightsheet_registration/` and tests in `tests/`; hatchling builds the
  wheel.
- ruff sets a line length of 120 and numpy-style docstrings, and exempts tests from the
  docstring and annotation rules. interrogate fails below 30%
  docstring coverage of `src`.
- mypy checks `src/aind_lightsheet_registration` in strict mode.
- Comments and docstrings say what the code does not: why it is written this way, and what breaks
  otherwise. They carry no history; the reason for a change belongs in its commit message.
- `scratch/` is gitignored and excluded from the linters, for work that is not kept.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
  `fix:`, `docs:`, `chore:`): an imperative title, and a body giving what changed and why. After CI
  passes on `main`, the bump workflow uses commitizen to bump the version from them and update
  `CHANGELOG.md`.
- A change a user would notice updates README.md in the same commit.
