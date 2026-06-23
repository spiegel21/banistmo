#!/bin/bash
# SessionStart hook for Claude Code on the web.
# Creates a per-project virtualenv and installs each project's dependencies so
# that tests and scripts work during remote sessions. Matches the repo
# convention (CLAUDE.md): each project under projects/ is self-contained with
# its own .venv and requirements.txt — no shared dependencies.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

setup_project() {
  project_dir="$1"
  shift
  extra_pkgs="$*"

  if [ ! -f "$project_dir/requirements.txt" ]; then
    echo "Skipping $project_dir (no requirements.txt)"
    return 0
  fi

  echo "==> Setting up $project_dir"
  # Idempotent: python venv creation is a no-op if .venv already exists.
  python3 -m venv "$project_dir/.venv"
  # shellcheck disable=SC1091
  . "$project_dir/.venv/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet -r "$project_dir/requirements.txt" $extra_pkgs
  deactivate
}

# Project 02 ships a pytest suite but does not list pytest in requirements.txt,
# so install it explicitly alongside the project deps.
setup_project "$ROOT/projects/02-pnl-dashboard" pytest
setup_project "$ROOT/projects/01-forex-strategy"

echo "Session start setup complete."
