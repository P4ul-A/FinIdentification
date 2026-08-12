#!/bin/zsh
set -e

cd -- "$(dirname "$0")"

REQUIRED_MAJOR=3
REQUIRED_MINOR=12
REQUIREMENTS="requirements.txt"
LAUNCHER_TTY="$(tty 2>/dev/null || true)"

pause_on_error() {
  echo ""
  echo "Fin Identification could not start. Copy the messages above if you need help."
  echo "Press any key to close this window."
  read -k 1
}
trap pause_on_error ERR

supported_python() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'
}

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This Finder launcher requires macOS."
  exit 1
fi

if [ -x "venv/bin/python" ]; then
  PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Python 3.12 was not found. Install Python 3.12, then try again."
    exit 1
  fi
  BASE_PYTHON="$(command -v python3.12)"
  if ! supported_python "$BASE_PYTHON"; then
    echo "The available Python is not version 3.12."
    exit 1
  fi
  echo "Creating the Fin Identification environment…"
  "$BASE_PYTHON" -m venv venv
  PYTHON="venv/bin/python"
fi

if ! supported_python "$PYTHON"; then
  echo "The local environment does not use Python 3.12."
  echo "Remove its venv folder and open this launcher again."
  exit 1
fi

if [ ! -f "$REQUIREMENTS" ]; then
  echo "Could not find $REQUIREMENTS."
  exit 1
fi

# Private model weights are intentionally not stored in GitHub. Keep the
# expected folders present after a fresh ZIP download/clone so separately
# supplied weights have an obvious destination.
mkdir -p model_recognition model_identification

if ! "$PYTHON" - <<'PY'
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, distribution

required = ("findetection_core", "PIL", "torch", "torchvision", "ultralytics")
if not all(importlib.util.find_spec(name) for name in required):
    raise SystemExit(1)
try:
    core = distribution("findetection-core")
except PackageNotFoundError:
    raise SystemExit(1)
if core.version != "0.2.0":
    raise SystemExit(1)
try:
    direct = json.loads(core.read_text("direct_url.json") or "{}")
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
vcs = direct.get("vcs_info", {})
if vcs.get("requested_revision") != "v0.2.0":
    raise SystemExit(1)
if "P4ul-A/FinDetection-MPS-Core" not in direct.get("url", ""):
    raise SystemExit(1)
PY
then
  echo "Installing required components. This can take several minutes the first time…"
  # Python 3.12 venvs may contain pip without setuptools. The shared
  # FinDetection core uses setuptools.build_meta and --no-build-isolation
  # deliberately reuses this environment's build toolchain.
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install --upgrade --no-build-isolation -r "$REQUIREMENTS"
fi

if ! "$PYTHON" - <<'PY'
from findetection_core import probe_runtime

available, detail = probe_runtime()
print(detail)
if not available:
    raise SystemExit(detail)
PY
then
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_FAST_MATH=1
export PYTORCH_MPS_PREFER_METAL=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fin-identification-matplotlib-${UID}"
mkdir -p "$MPLCONFIGDIR"

echo "Starting Fin Identification…"
"$PYTHON" finid_app.py

# A .command file opened from Finder runs in its own Terminal tab. Once the
# interface closes normally, close that tab as well. Error paths still stop at
# pause_on_error above so setup messages remain available to copy.
if [[ "$LAUNCHER_TTY" == /dev/* ]] && [[ "${TERM_PROGRAM:-}" == "Apple_Terminal" ]]; then
  "$PYTHON" - "$LAUNCHER_TTY" <<'PY'
import subprocess
import sys

terminal_tty = sys.argv[1]
apple_script = f'''
delay 0.3
tell application "Terminal"
  repeat with terminalWindow in windows
    repeat with terminalTab in tabs of terminalWindow
      if tty of terminalTab is "{terminal_tty}" then
        if (count tabs of terminalWindow) > 1 then
          close terminalTab
        else
          close terminalWindow
        end if
        return
      end if
    end repeat
  end repeat
end tell
'''
subprocess.Popen(
    ["osascript", "-e", apple_script],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
PY
fi
