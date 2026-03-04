"""
cleanup_audit.py — Scans project folder and categorises files.
Run: python cleanup_audit.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
import os, json
from pathlib import Path

ROOT = Path(__file__).parent
KEEP = {"main.py", "requirements.txt", "README.md", ".env", ".gitignore",
        "PERFORMANCE_GUIDE.md", "TESTING_GUIDE.md"}

def scan():
    results = {"keep": [], "orphan_candidates": []}
    for p in ROOT.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(ROOT))
            if p.name in KEEP or any(part.startswith(".") for part in p.parts):
                results["keep"].append(rel)
            else:
                results["orphan_candidates"].append(rel)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    scan()
