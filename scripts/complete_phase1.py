"""Reproduce the complete Phase 1 pipeline with inspectable per-step logs."""

from pathlib import Path
import argparse, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reuse-rf",
        action="store_true",
        help="Use existing RF models instead of fixed-parameter regeneration",
    )
    args = p.parse_args()
    env = os.environ.copy()
    env.update(
        MPLCONFIGDIR=str(ROOT / ".mplconfig"),
        XDG_CACHE_HOME=str(ROOT / ".cache"),
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    steps = [
        ["-m", "src.data.dataset_builder"],
        ["scripts/run_person_c.py", "--predict-only"],
    ]
    if not args.reuse_rf:
        steps.append(["scripts/run_person_d.py", "--save-models", "--resume"])
    steps += [
        ["scripts/predict_person_d.py"],
        ["scripts/verify_saved_boosters.py"],
        ["scripts/run_person_e.py"],
        ["scripts/run_phase1_tests.py"],
        ["scripts/build_notebook_03.py"],
        ["scripts/verify_phase1.py"],
        ["scripts/build_phase1_paper.py"],
    ]
    logs = ROOT / ".cache/phase1_runs"
    logs.mkdir(parents=True, exist_ok=True)
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] " + " ".join(step), flush=True)
        with (logs / f"{i:02d}.log").open("w") as out:
            subprocess.run(
                [sys.executable, *step],
                cwd=ROOT,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print("Phase 1 regeneration, verification and paper creation completed.")


if __name__ == "__main__":
    main()
