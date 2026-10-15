#!/bin/sh
# Run linters (default) or linters+checks without rebuilding the uv env.

set -u

main() {
  CHECKS=false
  PYTEST_ARGS=""
  SEEN_DASHDASH=false

  # Parse args: -c/--checks anywhere; collect pytest args (after -- or others)
  for arg in "$@"; do
    if [ "$SEEN_DASHDASH" = true ]; then
      PYTEST_ARGS="$PYTEST_ARGS $arg"
      continue
    fi
    case "$arg" in
      -c|--checks) CHECKS=true ;;
      --) SEEN_DASHDASH=true ;;
      *) PYTEST_ARGS="$PYTEST_ARGS $arg" ;;  # pass unknowns to pytest
    esac
  done

  ENV_PATH="${ENV_PATH:-.venv}"

  # Choose runner: active venv, local .venv, or uv (no sync)
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    BIN="$VIRTUAL_ENV/bin"
    run() { "$BIN/$@"; }
  elif [ -x "$ENV_PATH/bin/python" ]; then
    BIN="$ENV_PATH/bin"
    run() { "$BIN/$@"; }
  else
    PYVER="${PYVER:-}"
    if [ -z "$PYVER" ] && [ -f .python-version ]; then
      PYVER="$(tr -d ' \n' < .python-version)"
    fi
    export UV_PROJECT_ENVIRONMENT="$ENV_PATH"
    UV_ARGS="--frozen --no-sync"
    [ -n "$PYVER" ] && UV_ARGS="$UV_ARGS --python $PYVER"
    run() { uv run $UV_ARGS -- "$@"; }
  fi

  # Every step runs even after a failure so one pass reports all problems;
  # the exit status is nonzero if any step failed.
  FAILED=""
  step() {
    name="$1"
    shift
    echo "+ $name"
    run "$@" || FAILED="$FAILED
  $name"
  }

  step "ruff format" ruff format

  if [ "$CHECKS" = true ]; then
    step "ruff check" ruff check
    step "mypy" mypy
    step "interrogate -v src" interrogate -v src
    step "codespell --check-filenames" codespell --check-filenames
    # shellcheck disable=SC2086
    step "pytest --cov aind_lightsheet_registration$([ -n "$PYTEST_ARGS" ] && printf ' -- %s' "$PYTEST_ARGS")" \
      pytest --cov aind_lightsheet_registration $PYTEST_ARGS
  else
    echo "(checks skipped; pass -c or --checks to enable)"
  fi

  if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED" >&2
    return 1
  fi
  echo "All steps passed."
}

main "$@"
