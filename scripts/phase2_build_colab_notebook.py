"""Generate notebooks/phase2_colab_train.ipynb (Member 1).

The notebook is generated, not hand-edited: change this script and re-run it.
It only writes the notebook file; it does not execute any cell and does not touch
Phase 1 notebooks or builders.

    python scripts/phase2_build_colab_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/phase2_colab_train.ipynb"

TITLE = """\
# Phase 2: train on Colab (free tier)

Trains the models of one run file (`search` -> `pick` -> `final`) and writes everything to Google Drive.

**How to use**
1. `Runtime > Change runtime type > T4 GPU` (the GPU is not guaranteed on the free tier; the code also runs on CPU, only slower).
2. Edit `RUN_FILE` (and optionally `ONLY`) in the settings cell.
3. `Runtime > Run all`.
4. If Colab disconnects, open the notebook again and press **Run all again**. Finished work is skipped and interrupted runs resume from the last saved epoch (checkpoints are written to Drive after every epoch).

**Needs in Google Drive** (`MyDrive/Phase2_shared/inputs/`): `metr-la.h5` and `adj_mx.pkl` (sha256 is checked). Results land in `MyDrive/Phase2_shared/out/`.

**Spreading seeds across accounts** (after one account has finished search + pick): set `ONLY = ["dcrnn:43"]` or `["gat:44", "gat_identity:44"]` and run on another account that shares the same `Phase2_shared/out` folder.
"""

PRIVATE_REPO = """\
### If the repository is private
Either (a) add a GitHub token as a Colab secret named `GITHUB_TOKEN` (key icon in the left bar, enable notebook access) and the clone cell will use it, or (b) put a zip of the repository at `MyDrive/Phase2_shared/inputs/repo.zip` (zip the folder `traffic-flow-capstone`) and set `USE_REPO_ZIP = True` below.
"""

SETTINGS = '''\
# ---- settings (edit these) -------------------------------------------------
REPO_URL = "https://github.com/pankajbaid567/traffic-flow.git"
BRANCH = "main"                          # use e.g. "phase2/m2-gat" before the branches are merged
USE_REPO_ZIP = False                     # True: unzip inputs/repo.zip instead of cloning
RUN_FILE = "configs/phase2/runs_example.yaml"   # e.g. configs/phase2/runs_gat.yaml
ONLY = []                                # e.g. ["dcrnn:43"] to run only some seeds
DEVICE = "auto"                          # auto | cuda | cpu
DRIVE_SHARED = "/content/drive/MyDrive/Phase2_shared"
EXPECTED_SHA256 = {
    "metr-la.h5": "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05",
    "adj_mx.pkl": "a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3",
}
# -----------------------------------------------------------------------------
'''

MOUNT = '''\
import os, sys
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")      # safe to run again
    WORK = "/content"
else:
    DRIVE_SHARED = os.path.expanduser("~/Phase2_shared")   # local dry run
    WORK = os.path.expanduser("~/phase2_work")
    os.makedirs(WORK, exist_ok=True)
INPUTS = os.path.join(DRIVE_SHARED, "inputs")
OUT = os.path.join(DRIVE_SHARED, "out")
os.makedirs(OUT, exist_ok=True)
print("inputs:", INPUTS, "| outputs:", OUT)
'''

CLONE = '''\
import subprocess, shutil, zipfile

REPO_DIR = os.path.join(WORK, "traffic-flow-capstone")

def sh(cmd, cwd=None, check=True):
    print("$", cmd.replace(TOKEN, "***") if TOKEN else cmd)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check, text=True)

TOKEN = ""
if IN_COLAB:
    try:
        from google.colab import userdata
        TOKEN = userdata.get("GITHUB_TOKEN") or ""
    except Exception:
        TOKEN = ""

if USE_REPO_ZIP:
    if not os.path.isdir(REPO_DIR):
        with zipfile.ZipFile(os.path.join(INPUTS, "repo.zip")) as z:
            z.extractall(WORK)
else:
    url = REPO_URL.replace("https://", f"https://{TOKEN}@") if TOKEN else REPO_URL
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        sh(f"git clone --branch {BRANCH} {url} {REPO_DIR}")
    else:   # re-run after a disconnect: just update
        sh(f"git remote set-url origin {url}", cwd=REPO_DIR)
        sh("git fetch origin", cwd=REPO_DIR)
        sh(f"git checkout {BRANCH}", cwd=REPO_DIR)
        sh(f"git pull --ff-only origin {BRANCH}", cwd=REPO_DIR)
    sh("git remote set-url origin " + REPO_URL, cwd=REPO_DIR)   # do not keep the token in .git/config
sh("git log --oneline -1", cwd=REPO_DIR)
'''

INSTALL = '''\
sh(f"{sys.executable} -m pip install -q -r requirements-phase2.txt", cwd=REPO_DIR)
'''

INPUT_FILES = '''\
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

raw_dir = os.path.join(REPO_DIR, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)
for name, expected in EXPECTED_SHA256.items():
    dst = os.path.join(raw_dir, name)
    if not os.path.exists(dst):
        src = os.path.join(INPUTS, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} not found. Upload it to Drive (see the instructions above).")
        shutil.copy2(src, dst)
    actual = sha256(dst)
    if actual != expected:
        os.remove(dst)
        raise RuntimeError(f"{name}: sha256 mismatch ({actual}); the copy was removed. Re-upload the original file.")
    print("ok", name, actual[:12])
'''

ENVIRONMENT = '''\
os.environ["PHASE2_OUT"] = OUT
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    sh("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv", check=False)
else:
    print("No GPU. Training will be slow; try Runtime > Change runtime type > T4 GPU.")
only_flags = " ".join(f"--only {o}" for o in ONLY)
'''

RUN = '''\
# Each epoch prints seconds per epoch and an upper bound for the remaining time.
# Press "Run all" again after any disconnect: finished runs are skipped, others resume.
!cd {REPO_DIR} && python -u -m src.phase2.cli all --runs {RUN_FILE} --device {DEVICE} {only_flags}
'''

STATUS = '''\
import glob, json

def epoch_table():
    rows = []
    for log in sorted(glob.glob(os.path.join(OUT, "*", "**", "epochs.jsonl"), recursive=True)):
        lines = [json.loads(l) for l in open(log) if l.strip()]
        if not lines:
            continue
        run_dir = os.path.dirname(log)
        done = os.path.exists(os.path.join(run_dir, "done.json"))
        mean_s = sum(l["epoch_seconds"] for l in lines) / len(lines)
        rows.append((os.path.relpath(run_dir, OUT), len(lines), round(mean_s, 1), lines[-1]["best_val_mae"], "done" if done else "in progress"))
    print(f"{'run':<64}{'epochs':>7}{'s/epoch':>9}{'best val MAE':>14}  status")
    for r in rows:
        print(f"{r[0]:<64}{r[1]:>7}{r[2]:>9}{r[3]:>14}  {r[4]}")

epoch_table()
'''

OUTPUTS = '''\
pred_root = os.path.join(OUT, "predictions")
print("prediction files in", pred_root)
for path in sorted(glob.glob(os.path.join(pred_root, "*", "*"))):
    print(f"{os.path.relpath(path, pred_root):<48}{os.path.getsize(path) / 1e6:>8.1f} MB")
for meta in sorted(glob.glob(os.path.join(pred_root, "*", "seed*_meta.json"))):
    m = json.load(open(meta))
    print(f"{m['model_id']:<24} seed {m['seed']}: val MAE {m['best_val_mae']:.4f}, {m['epochs_run']} epochs, "
          f"{m['train_seconds'] / 60:.1f} min, stop={m['stop_reason']}, params={m['params']}, device={m['device']}")
'''


def build() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(TITLE),
        new_markdown_cell(PRIVATE_REPO),
        new_code_cell(SETTINGS),
        new_code_cell(MOUNT),
        new_code_cell(CLONE),
        new_code_cell(INSTALL),
        new_code_cell(INPUT_FILES),
        new_code_cell(ENVIRONMENT),
        new_markdown_cell("## Train (search, pick, final)\nSafe to run again after a disconnect."),
        new_code_cell(RUN),
        new_markdown_cell("## Status and outputs"),
        new_code_cell(STATUS),
        new_code_cell(OUTPUTS),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"phase2-cell-{index:02d}"
    notebook = new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    }
    return notebook


def main() -> int:
    notebook = build()
    nbformat.validate(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(notebook.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
