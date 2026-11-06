"""Exercise the actual Windows portable EXE across overlapping launches.

No USB operations: the extracted runtime opens a five-second PsychoPy window.
Each independent app uses a separate temporary profile, preserving user history.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil

if sys.platform != "win32":
    raise SystemExit("Run this test on Windows")
artifact = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[2]
folder = Path(tempfile.mkdtemp(prefix="spyder-portable-regression-"))
env = dict(os.environ)
env.pop("ELECTRON_RUN_AS_NODE", None)
launches = []


def launch(name, profile):
    log = (folder / f"{name}.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(artifact), f"--user-data-dir={profile}"],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    launches.append((process, log))
    return process


def find_backend(process):
    end = time.monotonic() + 120
    while time.monotonic() < end:
        if process.poll() is not None:
            raise AssertionError(f"App exited unexpectedly ({process.returncode})")
        for child in psutil.Process(process.pid).children(recursive=True):
            if child.name().lower() == "spyder-cal-backend.exe":
                executable = Path(child.exe())
                version = executable.parent / "_internal/psychopy/VERSION"
                assert version.read_text().strip() == "2026.2.5"
                return child, executable, version
        time.sleep(.2)
    raise AssertionError("Bundled backend did not start")


def stop(process, backend):
    if process.poll() is not None:
        return
    try:
        # Terminating Electron allows its NSIS parent to finish and clean up.
        gui = backend.parent()
        assert gui and gui.name().lower() == "spyder cal.exe"
        gui.terminate()
        process.wait(timeout=20)
    except psutil.NoSuchProcess:
        process.wait(timeout=20)


try:
    first = launch("first", folder / "profile-a")
    first_backend, first_executable, first_version = find_backend(first)

    # A repeated launch must focus the first window and close its own launcher.
    duplicate = launch("duplicate", folder / "profile-a")
    assert duplicate.wait(timeout=120) == 0
    assert first.poll() is None and first_backend.is_running()
    assert first_version.read_text().strip() == "2026.2.5"

    # Independent profiles exercise the extraction folders and cleanup directly.
    second = launch("second", folder / "profile-b")
    second_backend, second_executable, second_version = find_backend(second)
    assert first_executable.parent != second_executable.parent, "Portable launches share a runtime directory"
    stop(second, second_backend)
    assert not second_version.exists(), "The closed launch did not clean up"
    assert first_version.read_text().strip() == "2026.2.5", "Closing another launch deleted PsychoPy resources"
    assert first_backend.is_running()

    # Exercise a fresh PsychoPy import using the surviving extracted runtime.
    backend_env = {**env, "SPYDER_BACKEND": str(first_executable)}
    subprocess.run(["node", str(root / "desktop/test/frozen.cjs")],
                   cwd=root, env=backend_env, check=True, timeout=60)
    stop(first, first_backend)
    assert not first_version.exists(), "The first launch did not clean up"
    print(json.dumps({"passed": True, "artifact": str(artifact), "checks": [
        "duplicate launch preserves active runtime", "unique extraction directories",
        "closing another launch preserves PsychoPy VERSION", "extracted PsychoPy fullscreen window",
        "extracted JSON/CSV/PNG/SVG exports", "temporary runtime cleanup"], "logs": str(folder)}))
finally:
    for process, log in launches:
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log.close()
