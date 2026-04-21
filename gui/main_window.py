"""Top-level QMainWindow with the four tabs."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from .app_state import AppState
from .pages.controller_editor import ControllerEditorPage
from .pages.robot_builder import RobotBuilderPage
from .pages.simulation import SimulationPage
from .pages.track_builder import TrackBuilderPage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PySaac Sim")
        self.state = AppState()

        tabs = QTabWidget()
        tabs.addTab(SimulationPage(self.state), "Simulation / Train")
        tabs.addTab(TrackBuilderPage(self.state), "Track Builder")
        tabs.addTab(RobotBuilderPage(self.state), "Robot")
        tabs.addTab(ControllerEditorPage(self.state), "Controller")
        self.setCentralWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
