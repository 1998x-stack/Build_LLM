#!/usr/bin/env python3
"""run_all.py — 冒烟测试:运行每个课程 .py 并报告 PASS/FAIL; 非零退出表示有失败。"""
import glob, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {".git", "docs", "common", "tests", ".superpowers"}
SKIP = {"main.py", "modify_files.sh", "run_all.py"}

def _run(rel):
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, rel)],
                           cwd=ROOT, timeout=300, capture_output=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def main():
    files = []
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, ROOT)
        if any(p in EXCLUDE for p in rel.split(os.sep)) or os.path.basename(rel) in SKIP:
            continue
        files.append(rel)
    fails = []
    for rel in files:
        ok = _run(rel)
        print(("PASS" if ok else "FAIL"), rel)
        if not ok:
            fails.append(rel)
    if fails:
        print(f"\n{len(fails)} FAILED:", *fails, sep="\n  ")
        return 1
    print(f"\nAll {len(files)} lectures passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())