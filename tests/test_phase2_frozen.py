"""Unit tests for scripts/phase2_check_phase1_frozen.py (uses throw-away git repos)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_check", ROOT / "scripts/phase2_check_phase1_frozen.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


class FrozenCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        git(self.root, "init", "-q")
        (self.root / "phase1.py").write_text("print('phase1')\n")
        (self.root / "README.md").write_text("line1\n")
        (self.root / ".gitignore").write_text("a\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "base")
        git(self.root, "tag", "phase1-final")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        base = frozen.resolve_base(self.root)
        return base, frozen.find_violations(self.root, base)

    def test_clean_tree_is_ok(self):
        base, problems = self.check()
        self.assertEqual(base, "phase1-final")
        self.assertEqual(problems, [])

    def test_new_files_and_appends_are_ok(self):
        (self.root / "src").mkdir()
        (self.root / "src/new_phase2.py").write_text("x = 1\n")
        with (self.root / "README.md").open("a") as f:
            f.write("phase 2 pointer\n")
        with (self.root / ".gitignore").open("a") as f:
            f.write("b\n")
        self.assertEqual(self.check()[1], [])

    def test_modifying_phase1_file_is_flagged(self):
        (self.root / "phase1.py").write_text("print('changed')\n")
        problems = self.check()[1]
        self.assertEqual(len(problems), 1)
        self.assertIn("phase1.py", problems[0])

    def test_committed_change_is_flagged(self):
        (self.root / "phase1.py").write_text("print('changed')\n")
        git(self.root, "commit", "-q", "-am", "edit")
        self.assertTrue(any("phase1.py" in p for p in self.check()[1]))

    def test_deleting_phase1_file_is_flagged(self):
        (self.root / "phase1.py").unlink()
        self.assertTrue(any("phase1.py" in p and "deleted" in p for p in self.check()[1]))

    def test_rewriting_allowed_file_is_flagged(self):
        (self.root / "README.md").write_text("completely different\n")
        self.assertTrue(any("README.md" in p for p in self.check()[1]))

    def test_fallback_commit_when_tag_missing(self):
        git(self.root, "tag", "-d", "phase1-final")
        sha = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        with self.assertRaises(RuntimeError):
            frozen.resolve_base(self.root)  # neither the tag nor the fallback commit exists here
        self.assertEqual(frozen.resolve_base(self.root, sha), sha)

    def test_real_repo_baseline_resolves(self):
        base = frozen.resolve_base(ROOT)
        self.assertIn(base, {"phase1-final", "138aa57"})


if __name__ == "__main__":
    unittest.main()
