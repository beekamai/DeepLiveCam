"""Pick which face of a photo is the source when it shows several people."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from modules.gettext import _

MAX_SIDE = 900


class FacePickerDialog(QDialog):
    """Shows the photo with a box around every face; a click chooses one.
    ``chosen`` is the picked face's centre in image pixels, or ``None``."""

    def __init__(self, image: np.ndarray, faces: List[Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Select a face"))
        self.setModal(True)
        self._image = image
        self._faces = faces
        self._index = 0
        self.chosen: Optional[Tuple[float, float]] = None

        screen = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
        limit = min(MAX_SIDE, int(screen.height() * 0.7)) if screen else MAX_SIDE
        h, w = image.shape[:2]
        self._scale = min(1.0, limit / max(h, w))

        layout = QVBoxLayout(self)
        hint = QLabel(_("Several faces found. Click the one to use, then press Use."))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._view = QLabel()
        self._view.setCursor(Qt.CursorShape.PointingHandCursor)
        self._view.mousePressEvent = self._on_click
        layout.addWidget(self._view)
        buttons = QHBoxLayout()
        use = QPushButton(_("Use"))
        use.clicked.connect(self._accept)
        cancel = QPushButton(_("Cancel"))
        cancel.setObjectName("secondary")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(use)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._render()

    def _render(self) -> None:
        frame = self._image.copy()
        for i, face in enumerate(self._faces):
            x1, y1, x2, y2 = (int(v) for v in face.bbox[:4])
            colour = (60, 220, 60) if i == self._index else (60, 160, 240)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if i == self._index else 2)
        if self._scale < 1.0:
            frame = cv2.resize(frame, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._view.setPixmap(QPixmap.fromImage(image.copy()))
        self._view.setFixedSize(w, h)

    def _on_click(self, event) -> None:
        pos = event.position()
        x, y = pos.x() / self._scale, pos.y() / self._scale
        best, best_d = None, None
        for i, face in enumerate(self._faces):
            x1, y1, x2, y2 = face.bbox[:4]
            inside = x1 <= x <= x2 and y1 <= y <= y2
            d = ((x1 + x2) / 2 - x) ** 2 + ((y1 + y2) / 2 - y) ** 2
            if inside and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            self._index = best
            self._render()

    def _accept(self) -> None:
        face = self._faces[self._index]
        x1, y1, x2, y2 = face.bbox[:4]
        self.chosen = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        self.accept()
