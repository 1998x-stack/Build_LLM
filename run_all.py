#!/usr/bin/env python3
"""Run every lecture script and print actionable PASS/FAIL diagnostics."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
EXCLUDE = {
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    "common",
    "docs",
    "examples",
    "src",
    "tests",
    "tools",
}
SKIP = {"main.py", "modify_files.sh", "run_all.py"}


def _run(relative_path: str) -> tuple[bool, float, str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(ROOT / "src"),
        str(ROOT),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in pythonpath if part
    )
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / relative_path)],
            cwd=ROOT,
            env=env,
            timeout=120,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start
        return (
            result.returncode == 0,
            elapsed,
            result.stdout,
            result.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nTIMEOUT after 120s"
        return False, elapsed, stdout, stderr


def main() -> int:
    lecture_files: list[str] = []
    pattern = str(ROOT / "**" / "*.py")
    for path in sorted(glob.glob(pattern, recursive=True)):
        relative = os.path.relpath(path, ROOT)
        parts = Path(relative).parts
        if any(part in EXCLUDE for part in parts):
            continue
        if Path(relative).name in SKIP:
            continue
        lecture_files.append(relative)

    failures: list[str] = []
    for relative in lecture_files:
        ok, elapsed, stdout, stderr = _run(relative)
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} {elapsed:6.2f}s  {relative}")
        if not ok:
            failures.append(relative)
            if stdout.strip():
                print("  stdout:")
                print(stdout.rstrip())
            if stderr.strip():
                print("  stderr:")
                print(stderr.rstrip())

    if failures:
        print(f"\n{len(failures)} lecture(s) failed:")
        for relative in failures:
            print(f"  - {relative}")
        return 1

    print(f"\nAll {len(lecture_files)} lectures passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
