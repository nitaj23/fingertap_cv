import os
import sys

# Ensure the project root is importable regardless of pytest's invocation dir.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
