#!/usr/bin/env python3
"""Execute the two Person A notebooks with this Python interpreter.

Kernel specifications live in a temporary directory, so this command does not
install or change a user's Jupyter kernels. ``--in-process`` explicitly selects
socket-free IPython execution when the sandbox blocks a listening kernel. A
notebook is saved only after its entire execution and validation succeed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_data_understanding.ipynb",
    "02_data_quality.ipynb",
)


def execution_environment(method: str) -> dict:
    packages = {}
    for name in (
        "numpy", "pandas", "matplotlib", "PyYAML", "tables", "nbformat",
        "IPython", "nbclient", "ipykernel", "jupyter_client",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "method": method,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_timeout_seconds": 300,
    }


@contextmanager
def isolated_process_context(temporary: Path, cwd: Path):
    """Confine IPython state and restore environment/cwd on success or failure."""
    settings = {
        "IPYTHONDIR": str(temporary / "ipython"),
        "JUPYTER_RUNTIME_DIR": str(temporary / "runtime"),
    }
    old_environment = {key: os.environ.get(key) for key in settings}
    old_cwd = Path.cwd()
    os.environ.update(settings)
    try:
        os.chdir(cwd)
        yield
    finally:
        os.chdir(old_cwd)
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def cell_timeout(seconds: int = 300):
    """Apply the same per-cell limit as nbclient on this Unix command-line host."""
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError(f"Notebook cell exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def execute_in_process(notebook, cwd: Path):
    """Execute real cells through a fresh IPython shell, without kernel sockets.

    Captures MIME bundles and ordered streams directly from IPython's display
    publisher/display hook. This does not simulate or claim Jupyter execution.
    The caller must only save the returned notebook after full success.
    """
    import nbformat
    from IPython.core.displayhook import DisplayHook
    from IPython.core.displaypub import DisplayPublisher
    from IPython.core.interactiveshell import InteractiveShell
    from jupyter_client.jsonutil import json_clean
    from traitlets.config import Config

    class Recorder:
        def __init__(self):
            self.outputs = []
            self.execution_count = 0
            self.clear_on_next_output = False
            self.display_ids = {}

        def append(self, output):
            if self.clear_on_next_output:
                self.outputs.clear()
                self.clear_on_next_output = False
            self.outputs.append(output)

        def stream(self, name, text):
            if not text:
                return
            if self.clear_on_next_output:
                self.outputs.clear()
                self.clear_on_next_output = False
            if self.outputs and self.outputs[-1].output_type == "stream" and self.outputs[-1].name == name:
                self.outputs[-1].text += text
            else:
                self.append(nbformat.v4.new_output("stream", name=name, text=text))

    class OutputStream(io.TextIOBase):
        def __init__(self, recorder, name):
            self.recorder, self.name = recorder, name

        @property
        def encoding(self):
            return "utf-8"

        def write(self, text):
            self.recorder.stream(self.name, text)
            return len(text)

        def flush(self):
            pass

    class NotebookDisplayPublisher(DisplayPublisher):
        def publish(self, data, metadata=None, source=None, *, transient=None, update=False, **kwargs):
            recorder = self.shell._notebook_recorder
            data, metadata = json_clean(data), json_clean(metadata or {})
            display_id = (transient or {}).get("display_id")
            if update:
                for previous in recorder.display_ids.get(display_id, []):
                    previous.data, previous.metadata = data, metadata
                return
            output = nbformat.v4.new_output("display_data", data=data, metadata=metadata)
            recorder.append(output)
            if display_id:
                recorder.display_ids.setdefault(display_id, []).append(output)

        def clear_output(self, wait=False):
            recorder = self.shell._notebook_recorder
            if wait:
                recorder.clear_on_next_output = True
            else:
                recorder.outputs.clear()
                recorder.clear_on_next_output = False

    class NotebookDisplayHook(DisplayHook):
        def write_output_prompt(self):
            pass

        def write_format_data(self, format_dict, md_dict=None):
            recorder = self.shell._notebook_recorder
            recorder.append(nbformat.v4.new_output(
                "execute_result", execution_count=recorder.execution_count,
                data=json_clean(format_dict), metadata=json_clean(md_dict or {}),
            ))

    # New singleton/namespace for every notebook; no shared variables or history.
    InteractiveShell.clear_instance()
    recorder = Recorder()
    with tempfile.TemporaryDirectory(prefix="metr-la-in-process-") as temporary:
        with isolated_process_context(Path(temporary), Path(cwd)):
            config = Config()
            config.HistoryManager.hist_file = ":memory:"
            config.InteractiveShell.colors = "NoColor"
            shell = InteractiveShell.instance(
                config=config,
                ipython_dir=str(Path(temporary) / "ipython"),
                displayhook_class=NotebookDisplayHook,
            )
            shell._notebook_recorder = recorder
            shell.display_pub = NotebookDisplayPublisher(shell=shell)
            try:
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type != "code":
                        continue
                    cell.outputs = []
                    cell.execution_count = None
                    if not cell.source.strip():
                        continue
                    recorder.outputs = cell.outputs
                    recorder.clear_on_next_output = False
                    recorder.execution_count = shell.execution_count
                    cell.execution_count = recorder.execution_count
                    with redirect_stdout(OutputStream(recorder, "stdout")), redirect_stderr(OutputStream(recorder, "stderr")):
                        with cell_timeout():
                            result = shell.run_cell(cell.source, store_history=True, silent=False)
                    error = result.error_before_exec or result.error_in_exec
                    if error is not None:
                        cell.outputs.append(nbformat.v4.new_output(
                            "error", ename=type(error).__name__, evalue=str(error),
                            traceback=list(traceback.format_exception(type(error), error, error.__traceback__)),
                        ))
                        raise RuntimeError(
                            f"Code cell {index + 1} failed: {type(error).__name__}: {error}"
                        ) from error
            finally:
                shell.reset(new_session=False)
                shell.history_manager.end_session()
                if shell.history_manager.save_thread is not None:
                    shell.history_manager.save_thread.stop()
                shell.cleanup()
                InteractiveShell.clear_instance()

    notebook.metadata["execution_method"] = "in_process_ipython"
    notebook.metadata["execution_environment"] = execution_environment("in_process_ipython")
    notebook.metadata["execution_environment"]["reason"] = (
        "Explicit --in-process selection: the local sandbox denies kernel socket binding. "
        "Real cells executed by IPython InteractiveShell.run_cell; no Jupyter kernel was started."
    )
    nbformat.validate(notebook)
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-process", action="store_true",
        help="Execute real cells directly in IPython when sandbox socket restrictions prevent a Jupyter kernel.",
    )
    args = parser.parse_args()
    raw = ROOT / "data/raw/metr-la.h5"
    if not raw.is_file():
        print(
            "BLOCKED: data/raw/metr-la.h5 is absent. Acquire the real speed file "
            "and record its provenance with scripts/acquire_data.py first. "
            "See README.md and reports/data_sources.md. No notebooks were executed.",
            file=sys.stderr,
        )
        return 2

    required = ("nbformat", "IPython", "jupyter_client") if args.in_process else (
        "nbformat", "nbclient", "jupyter_client", "ipykernel",
    )
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print(
            f"BLOCKED: missing notebook dependencies: {', '.join(missing)}.\n"
            f"Install the project requirements using this interpreter:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
            "No notebooks were executed.",
            file=sys.stderr,
        )
        return 2

    import nbformat

    paths = [ROOT / "notebooks" / name for name in NOTEBOOKS]
    for path in paths:
        if not path.is_file():
            print(f"BLOCKED: notebook is absent: {path}", file=sys.stderr)
            return 2

    if args.in_process:
        print("Execution method: in-process IPython (socket-free; no Jupyter kernel).", flush=True)
        for path in paths:
            print(f"Executing {path.name} with {sys.executable}", flush=True)
            try:
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                executed = execute_in_process(notebook, ROOT)
                nbformat.write(executed, path)
            except Exception as error:
                print(
                    f"FAILED: {path.name}: {error}\n"
                    "This notebook's executed outputs were not saved. "
                    "Earlier successfully executed notebooks remain saved.",
                    file=sys.stderr,
                )
                return 1
            print(f"Saved successfully executed notebook: {path.name}", flush=True)
        return 0

    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    with tempfile.TemporaryDirectory(prefix="metr-la-notebook-kernel-") as temporary:
        kernel_root = Path(temporary) / "kernels"
        kernel_name = "person-a-local"
        specification = kernel_root / kernel_name
        specification.mkdir(parents=True)
        (specification / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Person A (current Python)",
                    "language": "python",
                    "env": {
                        "IPYTHONDIR": str(Path(temporary) / "ipython"),
                        "JUPYTER_RUNTIME_DIR": str(Path(temporary) / "runtime"),
                    },
                },
                indent=2,
            )
            + "\n"
        )
        specifications = KernelSpecManager(kernel_dirs=[str(kernel_root)])
        for path in paths:
            print(f"Executing {path.name} with {sys.executable}", flush=True)
            try:
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                manager = AsyncKernelManager(
                    kernel_name=kernel_name,
                    kernel_spec_manager=specifications,
                    connection_file=str(Path(temporary) / f"{path.stem}-connection.json"),
                )
                client = NotebookClient(
                    notebook,
                    km=manager,
                    timeout=300,
                    allow_errors=False,
                    force_raise_errors=True,
                    ipython_hist_file=":memory:",
                    resources={"metadata": {"path": str(ROOT)}},
                )
                # cleanup_kc=True also closes a caller-supplied KernelManager.
                executed = client.execute(cwd=str(ROOT), cleanup_kc=True)
                executed.metadata["execution_method"] = "nbclient"
                executed.metadata["execution_environment"] = execution_environment("nbclient")
                nbformat.validate(executed)
                nbformat.write(executed, path)
            except Exception as error:
                print(
                    f"FAILED: {path.name}: {error}\n"
                    "This notebook's executed outputs were not saved. "
                    "Earlier successfully executed notebooks remain saved.",
                    file=sys.stderr,
                )
                return 1
            print(f"Saved successfully executed notebook: {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
