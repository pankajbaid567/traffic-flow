"""Execute the full test suite and persist actual counts, skips and failures."""

from pathlib import Path
import sys, unittest, json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
suite = unittest.TestLoader().discover(str(ROOT / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
summary = {
    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "tests_run": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
    "successful": result.wasSuccessful(),
}
(ROOT / "data/metadata/phase1_test_results.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
raise SystemExit(not result.wasSuccessful())
