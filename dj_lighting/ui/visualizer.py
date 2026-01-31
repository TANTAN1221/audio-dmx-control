from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..qt_compat import QFrame, QPainter, QColor, QPen, QRect, no_pen, renderhint_antialiasing


class SimpleVisualizer(QFrame):
    def __init__(self, nbars: int = 64, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.nbars = nbars
        self._levels = np.zeros(nbars, dtype=np.float32)
        self._ema = np.ones(nbars, dtype=np.float32) * 1e-3
        self._alpha = 0.08
        self.setMinimumHeight(110)

    def apply_scale(self, s: float):
        self.setMinimumHeight(int(max(70, round(110 * s))))

    def update_from_spectrum(self, spectrum: Optional[List[float]]):
        if not spectrum:
            self._levels[:] = 0.0
            self.update()
            return

        src = np.array(spectrum, dtype=np.float32)
        src = np.maximum(src, 1e-6)
        idx = np.linspace(0, max(0, len(src) - 1), self.nbars).astype(np.int32)
        x = src[idx]

        self._ema = (1 - self._alpha) * self._ema + self._alpha * x
        self._levels = np.clip(x / (self._ema * 3.0), 0.0, 1.0)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(renderhint_antialiasing())

        r = self.rect().adjusted(12, 12, -12, -12)
        p.fillRect(r, QColor(15, 17, 20, 190))

        nb = self.nbars
        gap = 2
        bar_w = max(2, int((r.width() - gap * (nb - 1)) / nb))
        max_h = r.height()

        p.setPen(QPen(QColor(46, 51, 58, 160)))
        for frac in (0.25, 0.5, 0.75):
            y = r.bottom() - int(max_h * frac)
            p.drawLine(r.left(), y, r.right(), y)

        p.setPen(no_pen())
        for i in range(nb):
            lvl = float(self._levels[i])
            bh = int(max_h * lvl)
            x = r.left() + i * (bar_w + gap)
            y = r.bottom() - bh
            p.fillRect(QRect(x, y, bar_w, bh), QColor(91, 137, 184, 200))

        p.end()
