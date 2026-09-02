"""
Optional launcher. Kept for convenience — main.py now runs on its own, so
either of these works:

    python main.py
    python run_server.py

Right-click either file in PyCharm and choose Run, or use the green play
button. This file must sit in the same folder as main.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import serve  # noqa: E402

if __name__ == "__main__":
    serve()
