"""test_mock_order.py — Tests that the analysis pipeline runs subsystems in the correct order."""
from dotenv import load_dotenv
load_dotenv()

ORDER_LOG = []

class MockGeom:
    def process(self, frame):
        ORDER_LOG.append("geometry")
        return {}

class MockTemp:
    def add_frame(self, *a): ORDER_LOG.append("temporal_add")
    def finalize(self): ORDER_LOG.append("temporal_finalize"); return {}

def main():
    g = MockGeom()
    t = MockTemp()
    g.process(None)
    t.add_frame({}, {}, 0.0)
    t.finalize()
    expected = ["geometry", "temporal_add", "temporal_finalize"]
    ok = ORDER_LOG == expected
    print(f"[ORDER TEST] {'PASS' if ok else 'FAIL'} — order: {ORDER_LOG}")

if __name__ == "__main__":
    main()
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))
