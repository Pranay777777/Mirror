"""find_singular_fallback.py — Finds code paths that fall back to a singular/default value."""
import ast, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

FALLBACK_PATTERNS = ["0.5", "0.0", "None", "\"neutral\"", "'neutral'"]

def scan_file(path):
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            # Ternary expression — possible fallback
            pass  # placeholder for deeper analysis
    # Simple text search
    for i, line in enumerate(src.splitlines(), 1):
        for pat in FALLBACK_PATTERNS:
            if f"if {pat}" in line or f"or {pat}" in line or f"else {pat}" in line:
                print(f"  {path.relative_to(ROOT)}:{i}: {line.strip()}")

if __name__ == "__main__":
    print("[FIND SINGULAR FALLBACKS]")
    for py in ROOT.rglob("*.py"):
        if ".venv" not in str(py):
            scan_file(py)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
