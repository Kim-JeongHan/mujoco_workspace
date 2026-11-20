"""Bridge Colab's Python execution API to the uploaded shell worker."""

import os
import re
import subprocess
import sys

run_id = os.environ["MLAB_RUN_ID"]
if not re.fullmatch(r"(?:flow|mse)-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", run_id):
    raise ValueError("Invalid run identifier")

environment = os.environ.copy()
environment["MLAB_PYTHON"] = sys.executable
process = subprocess.Popen(
    ["bash", f"/content/{run_id}-worker.sh"],
    env=environment,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert process.stdout is not None
for line in process.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
if process.wait():
    raise RuntimeError(f"Colab BC worker exited with code {process.returncode}")
