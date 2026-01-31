from __future__ import annotations

from ..qt_compat import QFrame, QVBoxLayout, QLabel


class Panel(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(14, 14, 14, 14)
        self._outer.setSpacing(10)

        if title:
            t = QLabel(title)
            t.setObjectName("PanelTitle")
            self._outer.addWidget(t)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(10)
        self._outer.addLayout(self.body)

    def apply_scale(self, s: float):
        m = int(max(6, round(14 * s)))
        sp = int(max(6, round(10 * s)))
        self._outer.setContentsMargins(m, m, m, m)
        self._outer.setSpacing(sp)
        self.body.setSpacing(sp)
