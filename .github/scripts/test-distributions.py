"""Smoke-test installed distributions outside the source checkout."""

import os
import subprocess
import sys
import venv
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    dist = Path("dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels or not sdists:
        raise SystemExit("Build both wheel and sdist with make dist first")

    for archive in wheels + sdists:
        with TemporaryDirectory(prefix="ffn-dist-") as directory:
            environment = Path(directory) / "venv"
            venv.EnvBuilder().create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run(
                [sys.executable, "-m", "uv", "pip", "install", "--python", str(python), str(archive.resolve())],
                check=True,
                cwd=directory,
            )
            subprocess.run(
                [str(python), "-I", "-c", "import ffn; import pandas as pd; assert pd.Series([100.0, 110.0]).to_returns().iloc[1] > 0"],
                check=True,
                cwd=directory,
            )
        print(f"Passed: {archive}", flush=True)


if __name__ == "__main__":
    main()
