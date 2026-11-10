"""Check that Phase 1 files were not changed by Phase 2 work.

Compares the working tree (so staged, unstaged and committed changes all count)
with git tag ``phase1-final``. If that tag does not exist yet, the Phase 1
last published Phase 1 commit ``138aa57`` ("update redme", on top of the Phase 1
completion commit ``f1e4f22``) is used instead.

A "Phase 1 path" is any file tracked at the baseline. Adding new files is always
fine. Modifying or deleting a Phase 1 path is a violation, with two exceptions
that are allowed only if the change is append-only (no deleted lines):
``.gitignore`` and ``README.md``.

Usage (from anywhere inside the repo)::

    python scripts/phase2_check_phase1_frozen.py [--base REF]

Prints ``OK`` and exits 0, or lists the violations and exits 1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "phase1-final"
FALLBACK_COMMIT = "138aa57"
APPEND_ONLY_ALLOWED = {".gitignore", "README.md"}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def resolve_base(root: Path, base: str | None = None) -> str:
    """Return the baseline ref: explicit ``base``, else the tag, else the fallback commit."""
    candidates = [base] if base else [TAG, FALLBACK_COMMIT]
    for ref in candidates:
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return ref
    raise RuntimeError(f"none of the baseline refs exist: {candidates}")


def find_violations(root: Path, base: str) -> list[str]:
    """List human-readable violations of the frozen-Phase-1 rule."""
    phase1_paths = set(_git(root, "ls-tree", "-r", "--name-only", base).splitlines())
    # --no-renames: a rename becomes delete + add, so the old Phase 1 path is flagged.
    status_lines = _git(root, "diff", "--name-status", "--no-renames", base).splitlines()
    numstat = {}
    for line in _git(root, "diff", "--numstat", "--no-renames", base).splitlines():
        added, deleted, path = line.split("\t", 2)
        numstat[path] = (added, deleted)

    problems: list[str] = []
    for line in status_lines:
        status, path = line.split("\t", 1)
        if path not in phase1_paths:
            continue  # new Phase 2 file
        if path in APPEND_ONLY_ALLOWED and status == "M":
            added, deleted = numstat.get(path, ("0", "0"))
            if deleted not in ("0", "-"):
                problems.append(f"{path}: {deleted} line(s) removed or changed (only appends are allowed)")
            continue
        problems.append(f"{path}: {'deleted' if status == 'D' else 'modified'} (Phase 1 file)")
    return sorted(problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="baseline ref (default: tag phase1-final, else commit 138aa57)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args(argv)
    try:
        base = resolve_base(args.root, args.base)
        problems = find_violations(args.root, base)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if problems:
        print(f"Phase 1 files changed since {base}:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print(f"OK (no Phase 1 file changed since {base})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
