from __future__ import annotations

import numpy as np

from ..qt_compat import (
    QFrame, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush,
    QRect, QPointF, no_pen, renderhint_antialiasing
)


class SimulationWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self._base_min_h = 360
        self.setMinimumHeight(self._base_min_h)

        self._intensity = 0
        self._rgb = (0, 120, 255)
        self._strobe_on = True
        self._fx_name = "Off"
        self._pan = 270
        self._tilt = 90

    def apply_scale(self, s: float):
        self.setMinimumHeight(int(max(220, round(self._base_min_h * s))))

    def set_state(self, intensity_0_100: int, r: int, g: int, b: int, strobe_on: bool, fx_name: str, pan: int, tilt: int):
        self._intensity = int(np.clip(intensity_0_100, 0, 100))
        self._rgb = (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255)))
        self._strobe_on = bool(strobe_on)
        self._fx_name = (fx_name or "Off")
        self._pan = int(np.clip(pan, 0, 540))
        self._tilt = int(np.clip(tilt, 0, 180))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(renderhint_antialiasing())

        R = self.rect().adjusted(14, 14, -14, -14)
        p.fillRect(R, QColor(10, 12, 16, 230))

        # header
        p.setPen(QPen(QColor(214, 217, 222, 180), 1))
        p.drawText(
            R.left() + 10,
            R.top() + 22,
            f"Simulation — FX: {self._fx_name}   |   Pan: {self._pan}°   Tilt: {self._tilt}°"
        )

        inner = R.adjusted(8, 32, -8, -10)

        # fixture position
        fx_x = inner.center().x()
        fx_y = inner.bottom()

        # map pan/tilt to a target point
        pan_norm = (self._pan % 540) / 540.0
        tx = inner.left() + pan_norm * inner.width()

        tilt_norm = float(np.clip(self._tilt / 180.0, 0.0, 1.0))
        ty = inner.bottom() - tilt_norm * (inner.height() * 0.92)

        # draw fixture head
        p.setPen(no_pen())
        p.setBrush(QColor(32, 36, 42, 255))
        p.drawEllipse(QRect(int(fx_x - 14), int(fx_y - 16), 28, 28))
        p.setBrush(QColor(70, 78, 90, 220))
        p.drawEllipse(QRect(int(fx_x - 7), int(fx_y - 9), 14, 14))

        # if off or strobe off, stop after drawing fixture
        if self._intensity <= 0 or not self._strobe_on:
            p.end()
            return

        r, g, b = self._rgb
        strength = self._intensity / 100.0
        a_beam = int(np.clip(200 * strength, 0, 220))
        a_hot = int(np.clip(220 * strength, 0, 240))

        # beam spread
        dist = float(np.hypot(tx - fx_x, ty - fx_y))
        spread = float(np.clip(0.10 * dist + 25.0, 35.0, 220.0))
        half = spread * 0.5

        # beam polygon
        poly = QPolygonF([
            QPointF(fx_x - 10, fx_y - 6),
            QPointF(fx_x + 10, fx_y - 6),
            QPointF(tx + half, ty),
            QPointF(tx - half, ty),
        ])

        # beam gradient
        grad = QLinearGradient(QPointF(fx_x, fx_y), QPointF(tx, ty))
        grad.setColorAt(0.0, QColor(r, g, b, int(a_beam * 0.05)))
        grad.setColorAt(0.2, QColor(r, g, b, int(a_beam * 0.35)))
        grad.setColorAt(1.0, QColor(r, g, b, 0))

        p.setBrush(QBrush(grad))
        p.drawPolygon(poly)

        # hotspot
        spot_r = float(np.clip(22.0 + 0.08 * spread, 28.0, 120.0))
        rg = QRadialGradient(QPointF(tx, ty), spot_r)
        rg.setColorAt(0.0, QColor(r, g, b, a_hot))
        rg.setColorAt(0.6, QColor(r, g, b, int(a_hot * 0.35)))
        rg.setColorAt(1.0, QColor(r, g, b, 0))
        p.setBrush(QBrush(rg))
        p.drawEllipse(QRect(int(tx - spot_r), int(ty - spot_r), int(spot_r * 2), int(spot_r * 2)))

        # subtle floor glow
        floor_r = float(np.clip(spot_r * 2.2, 80.0, 260.0))
        fg = QRadialGradient(QPointF(tx, inner.bottom()), floor_r)
        fg.setColorAt(0.0, QColor(r, g, b, int(a_hot * 0.18)))
        fg.setColorAt(1.0, QColor(r, g, b, 0))
        p.setBrush(QBrush(fg))
        p.drawEllipse(QRect(int(tx - floor_r), int(inner.bottom() - floor_r * 0.55),
                            int(floor_r * 2), int(floor_r * 1.1)))

        p.end()
