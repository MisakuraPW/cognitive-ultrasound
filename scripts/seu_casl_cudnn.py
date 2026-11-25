"""Find a compatible existing cuDNN for the SEU CASL job before requesting a GPU."""

import ctypes
import json
import os
import site
import subprocess
import sys
from pathlib import Path


def candidates(root):
    explicit = os.environ.get("CASL_SEU_CUDNN_LIB")
    if explicit:
        yield Path(explicit)
        return
    prefix = Path(sys.prefix)
    yield from prefix.glob("lib/python*/site-packages/nvidia/cudnn/lib")
    yield Path(site.getusersitepackages()) / "nvidia/cudnn/lib"
    yield prefix / "lib"
    yield from root.glob("runtime/cudnn*/lib")
    yield from root.glob("*/nvidia/cudnn/lib")
    yield from root.glob("*/cudnn*/lib")
    # Older setup jobs may already have recorded the working library path.
    for report in sorted((root / "logs").glob("*gpu*.json")):
        if not report.is_file() or report.stat().st_size > 1_000_000:
            continue
        try:
            record = json.loads(report.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("passed") is True:
            yield from path_list(record.get("library_path", ""))
    yield from path_list(os.environ.get("LD_LIBRARY_PATH", ""))
    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDNN_ROOT"):
        if os.environ.get(variable):
            base = Path(os.environ[variable])
            yield base / "lib64"
            yield base / "lib"
    for base in (root, Path("/seu_share/apps"), Path("/usr/local")):
        for pattern in ("*cuda*12*/lib64", "*cuda*12*/targets/x86_64-linux/lib",
                        "*cudnn*/lib", "*cudnn*/lib64"):
            yield from sorted(base.glob(pattern), reverse=True)
    yield Path("/usr/lib/x86_64-linux-gnu")


def path_list(value):
    return (Path(p) for p in value.split(":") if p.startswith("/"))


def find_library(root):
    checked = set()
    for directory in candidates(root):
        directory = directory.resolve()
        if directory in checked or not (directory / "libcudnn.so.9").is_file():
            continue
        checked.add(directory)
        # Separate processes keep the dynamic loader from reusing a previous candidate.
        env = dict(os.environ, LD_LIBRARY_PATH=str(directory) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        try:
            result = subprocess.run(
                [sys.executable, __file__, "--version", str(directory)], env=env,
                capture_output=True, text=True, timeout=8, check=True,
            )
            version = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            print(f"cuDNN could not load in {directory}", file=sys.stderr)
            continue
        print(f"cuDNN {version} in {directory}", file=sys.stderr)
        if 90800 <= version < 100000 and (directory / "libcudnn_ops.so.9").is_file():
            return directory
    raise SystemExit("No loadable cuDNN 9.8+ (major 9) with runtime components found. "
                     "Set CASL_SEU_CUDNN_LIB or prepare the project runtime before submitting.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--version":
        lib = ctypes.CDLL(str(Path(sys.argv[2]) / "libcudnn.so.9"))
        lib.cudnnGetVersion.restype = ctypes.c_size_t
        print(lib.cudnnGetVersion())
    else:
        print(find_library(Path(sys.argv[1])))
