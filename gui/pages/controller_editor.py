"""Controller editor: paste raw C firmware or select a .c file, compile via
CController, register under a name so the Simulation page can use it."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ...control.c_bridge import CController
from ..app_state import AppState


_STARTER = '''\
/* PySaacSim controller template. Paste your firmware Robot() body into robot_tick().
 * See PySaacSim/examples/controller_template.c for a full wall-follower example. */

#include "hal.h"

void robot_init(void) {
    /* runs once at episode start */
}

void robot_tick(void) {
    /* runs every 80 ms. Read: DistanceRaw, L_DistanceRaw, Distance2, L_Distance2
     * Write: CAN_SetMotors(duty_l, duty_r, servo_count) */
    CAN_SetMotors(0, 0, 3000);   /* stop, wheels centered */
}
'''


_QUICK_PASTE_PROMPT = '''\
/* QUICK-PASTE MODE
 *
 * Paste the BODY of your firmware Robot()'s while(1) loop below (just the
 * lines between while(1) { ... }). Everything you need is already provided
 * by PySaacSim:
 *   Inputs  (set by Python each tick):
 *     DistanceRaw, L_DistanceRaw   -- IR mm (right / left)
 *     Distance2,   L_Distance2     -- TFLuna mm (right / left)
 *     Distance_Center              -- TFLuna center, mm
 *   Persistent state already declared:
 *     prevError, prevE_A, prevTime, elapsed
 *   Output:
 *     CAN_SetMotors(duty_l, duty_r, servo_count)
 *   OS_bWait(&TFLuna3Ready), OS_bWait(&TFLuna2Ready) are no-ops and safe
 *   to keep; __disable_irq/__enable_irq are no-ops too.
 *
 * Put any helper functions / LUTs / static variables ABOVE the
 *   // === LOOP BODY ===
 * marker. Paste the actual while(1) body BELOW it.
 */

#include <stdint.h>
#include "hal.h"

#define angle_ref 5
#define dist_ref  130
#define kp_d 1
#define kd_d 4
#define kp_a 5
#define kd_a 2

/* paste your helpers / LUTs here -------------------------------------------*/

// === LOOP BODY ===
/* paste the body of your while(1) { ... } here ---------------------------- */
'''


def _expand_quick_paste(src: str) -> str:
    """If src uses the quick-paste marker '// === LOOP BODY ===', split it
    into header + body and synthesize a compilable file with robot_init()
    and robot_tick(). Otherwise return src unchanged."""
    marker = "// === LOOP BODY ==="
    if marker not in src:
        return src
    header, _, body = src.partition(marker)
    return (
        header
        + "\nvoid robot_init(void) {\n"
        "    prevError = 0; prevE_A = 0; prevTime = 0;\n"
        "    elapsed = 0; Running = 1;\n"
        "}\n\n"
        "void robot_tick(void) {\n"
        "    elapsed  = OS_MsTime() - prevTime;\n"
        "    prevTime = OS_MsTime();\n"
        + body
        + "\n}\n"
    )


class ControllerEditorPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- left: editor ------------------------------------------------
        left = QWidget()
        v = QVBoxLayout(left)

        v.addWidget(self._build_source_group())
        v.addWidget(self._build_compile_group())

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Compiler output will appear here.")
        f = QFont("Consolas"); f.setPointSize(10)
        self.log.setFont(f)
        v.addWidget(self.log, 1)
        splitter.addWidget(left)

        # --- right: registered controllers ------------------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Registered controllers:"))
        self.registered = QListWidget()
        rv.addWidget(self.registered, 1)
        btn_row = QHBoxLayout()
        btn_remove = QPushButton("Remove selected")
        btn_remove.clicked.connect(self._remove_selected)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_remove)
        rv.addLayout(btn_row)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1000, 350])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        state.controllers_changed.connect(self._refresh_registered)
        self._refresh_registered()

    # ---- groups ----------------------------------------------------------

    def _build_source_group(self) -> QGroupBox:
        g = QGroupBox("C source")
        lay = QVBoxLayout(g)

        top = QHBoxLayout()
        top.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit("my_ctrl")
        top.addWidget(self.name_edit, 1)
        btn_load = QPushButton("Load from file…")
        btn_load.clicked.connect(self._load_file)
        btn_template = QPushButton("Load template")
        btn_template.clicked.connect(self._load_template)
        btn_quick = QPushButton("Quick-paste skeleton")
        btn_quick.setToolTip("Load a skeleton where you just paste the body "
                             "of your firmware while(1) loop.")
        btn_quick.clicked.connect(self._load_quick_paste)
        btn_example = QPushButton("Load wall-follower example")
        btn_example.clicked.connect(self._load_example)
        top.addWidget(btn_load)
        top.addWidget(btn_template)
        top.addWidget(btn_quick)
        top.addWidget(btn_example)
        lay.addLayout(top)

        self.editor = QPlainTextEdit()
        f = QFont("Consolas"); f.setPointSize(11)
        self.editor.setFont(f)
        self.editor.setPlainText(_STARTER)
        lay.addWidget(self.editor, 1)
        return g

    def _build_compile_group(self) -> QGroupBox:
        g = QGroupBox("Compile & Register")
        lay = QHBoxLayout(g)
        btn = QPushButton("Compile + Register")
        btn.clicked.connect(self._compile)
        lay.addWidget(btn)
        self.status = QLabel("idle")
        self.status.setStyleSheet("color: #666;")
        lay.addWidget(self.status, 1)
        return g

    # ---- handlers --------------------------------------------------------

    def _load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load C file", "",
                                              "C source (*.c *.h);;All (*)")
        if not path:
            return
        try:
            text = Path(path).read_text()
        except Exception as e:
            QMessageBox.warning(self, "Read failed", str(e))
            return
        self.editor.setPlainText(text)
        # Auto-populate the name field if still at default.
        if self.name_edit.text() in ("", "my_ctrl"):
            self.name_edit.setText(Path(path).stem)

    def _load_template(self) -> None:
        self.editor.setPlainText(_STARTER)

    def _load_quick_paste(self) -> None:
        self.editor.setPlainText(_QUICK_PASTE_PROMPT)

    def _load_example(self) -> None:
        example = Path(__file__).resolve().parents[2] / "examples" / "controller_template.c"
        if not example.exists():
            QMessageBox.warning(self, "Missing example",
                                f"Could not find {example}.")
            return
        self.editor.setPlainText(example.read_text())

    def _compile(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required",
                                "Give the controller a name so you can pick it on the sim page.")
            return
        self.log.clear()
        self.status.setText("compiling…")
        src_text = self.editor.toPlainText()
        src_text = _expand_quick_paste(src_text)
        # Write editor contents to a temp .c file and hand to CController.
        tmp = Path(tempfile.mkdtemp(prefix="pysim_ctrl_")) / f"{name}.c"
        tmp.write_text(src_text)
        try:
            ctrl = CController(tmp)
        except Exception as e:
            self.status.setText("compile failed")
            self.log.setPlainText(str(e))
            return
        self.state.register_controller(name, ctrl)
        self.status.setText(f"registered '{name}'")
        self.log.appendPlainText(f"Successfully compiled and registered '{name}'.")

    def _refresh_registered(self) -> None:
        self.registered.clear()
        for name in self.state.controllers.keys():
            self.registered.addItem(name)

    def _remove_selected(self) -> None:
        items = self.registered.selectedItems()
        if not items:
            return
        name = items[0].text()
        self.state.unregister_controller(name)
