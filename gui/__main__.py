"""Launch the PySaacSim GUI.

Run:
    python -m PySaacSim.gui
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PySaac Sim")
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
