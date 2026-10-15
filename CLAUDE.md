# CLAUDE.md

Instructions for coding agents in this repository. [CONTRIBUTING.md](CONTRIBUTING.md) holds setup,
checks, tests and conventions, and applies to agents as to people: read it before running or
changing anything. [README.md](README.md) documents use.

- Run every command through `uv`; never bare `pip` or `python`.
- Before calling a change done, run `./scripts/run_linters_and_checks.sh -c`.
- Keep this file to what only an agent needs; development practice belongs in CONTRIBUTING.md and
  usage in README.md.
