"""Models dialog: what is downloaded, what is not, fetch ahead of time."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from modules import model_library
from modules.gettext import _

KIND_LABELS = {
    "detector": "Detector",
    "swapper": "Face swapper",
    "enhancer": "Face enhancer",
    "mask": "Mask",
}


class _Downloader(QThread):
    progress = Signal(str)
    finished_entry = Signal(int, bool)
    all_done = Signal()

    def __init__(self, entries: List[model_library.Entry], indices: List[int]) -> None:
        super().__init__()
        self._entries = entries
        self._indices = indices
        self.cancelled = False

    def run(self) -> None:
        for index in self._indices:
            if self.cancelled:
                break
            ok = model_library.download(self._entries[index], self.progress.emit)
            self.finished_entry.emit(index, ok)
        self.all_done.emit()


class ModelLibraryDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Models"))
        self.setModal(False)
        self.resize(720, 460)
        self._entries = model_library.catalog()
        self._worker: Optional[_Downloader] = None

        layout = QVBoxLayout(self)
        hint = QLabel(_("Every model downloads itself the first time it is used; fetch it here "
                        "instead so the first Live does not wait. Files go to the models folder."))
        hint.setWordWrap(True)
        hint.setObjectName("statusLabel")
        layout.addWidget(hint)

        self._table = QTableWidget(len(self._entries), 5)
        self._table.setHorizontalHeaderLabels([_("Model"), _("Type"), _("Size"), _("Status"), ""])
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(36)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._table, 1)
        self._buttons: List[QPushButton] = []
        for row, entry in enumerate(self._entries):
            self._table.setItem(row, 0, QTableWidgetItem(entry.label))
            self._table.setItem(row, 1, QTableWidgetItem(_(KIND_LABELS.get(entry.kind, entry.kind))))
            size = QTableWidgetItem(entry.size_text)
            size.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 2, size)
            self._table.setItem(row, 3, QTableWidgetItem(""))
            button = QPushButton(_("Download"))
            button.setObjectName("secondary")
            button.clicked.connect(lambda _checked=False, r=row: self._download([r]))
            self._table.setCellWidget(row, 4, button)
            self._buttons.append(button)
            self._refresh_row(row)

        bottom = QHBoxLayout()
        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        bottom.addWidget(self._status, 1)
        self._all = QPushButton(_("Download all missing"))
        self._all.clicked.connect(self._download_missing)
        bottom.addWidget(self._all)
        close = QPushButton(_("Close"))
        close.setObjectName("secondary")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        layout.addLayout(bottom)
        self._update_all_button()

    def _refresh_row(self, row: int) -> None:
        entry = self._entries[row]
        self._table.item(row, 3).setText(_("downloaded") if entry.present else _("not downloaded"))
        self._buttons[row].setEnabled(not entry.present and self._worker is None)
        self._buttons[row].setText(_("Ready") if entry.present else _("Download"))

    def _update_all_button(self) -> None:
        missing = [i for i, e in enumerate(self._entries) if not e.present]
        total = sum(e.size or 0 for e in self._entries if not e.present)
        self._all.setEnabled(bool(missing) and self._worker is None)
        if missing:
            mb = total / (1024 * 1024)
            size = f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
            self._all.setText(_("Download all missing ({size})").format(size=size))
        else:
            self._all.setText(_("Everything is downloaded"))

    def _download_missing(self) -> None:
        self._download([i for i, e in enumerate(self._entries) if not e.present])

    def _download(self, indices: List[int]) -> None:
        if self._worker is not None or not indices:
            return
        self._worker = _Downloader(self._entries, indices)
        self._worker.progress.connect(self._status.setText)
        self._worker.finished_entry.connect(self._on_entry_done)
        self._worker.all_done.connect(self._on_all_done)
        for button in self._buttons:
            button.setEnabled(False)
        self._all.setEnabled(False)
        self._worker.start()

    def _on_entry_done(self, row: int, ok: bool) -> None:
        if not ok:
            self._status.setText(_("Could not download {name} — see the console.")
                                 .format(name=self._entries[row].label))
        self._refresh_row(row)

    def _on_all_done(self) -> None:
        self._worker = None
        for row in range(len(self._entries)):
            self._refresh_row(row)
        self._update_all_button()
        if all(e.present for e in self._entries):
            self._status.setText(_("Everything is downloaded"))
        elif not self._status.text().startswith(_("Could not download")):
            self._status.setText("")

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancelled = True
            self._worker.wait(2000)
        event.accept()
