from __future__ import annotations

from typing import Tuple

import numpy as np

from ..qt_compat import (
    QAbstractButton, QToolButton, QDrag, QMimeData, Qt, QPoint,
    QPainter, QColor, QPen, QRect, QLinearGradient, QBrush,
    QFrame, QVBoxLayout, QLabel, QSlider, QSizePolicy,
    cursor_pointing_hand, renderhint_antialiasing, align_hcenter, orientation_vertical
)


class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(cursor_pointing_hand())
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._base_w = 56
        self._base_h = 28
        self.apply_scale(1.0)

    def apply_scale(self, s: float):
        w = int(max(40, round(self._base_w * s)))
        h = int(max(22, round(self._base_h * s)))
        self.setFixedSize(w, h)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(renderhint_antialiasing())

        r = self.rect().adjusted(1, 1, -1, -1)
        checked = self.isChecked()

        track_off = QColor(10, 12, 16, 255)
        track_on = QColor(34, 58, 86, 255)
        border = QColor(50, 56, 68, 255)
        border_on = QColor(91, 137, 184, 255)

        p.setPen(QPen(border_on if checked else border, 1))
        p.setBrush(track_on if checked else track_off)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)

        thumb_r = max(10, r.height() - 6)
        y = r.top() + 3
        x = (r.right() - 3 - thumb_r) if checked else (r.left() + 3)

        grad = QLinearGradient(x, y, x, y + thumb_r)
        grad.setColorAt(0.0, QColor(230, 235, 242, 255))
        grad.setColorAt(1.0, QColor(170, 178, 188, 255))

        p.setPen(QPen(QColor(40, 45, 54, 180), 1))
        p.setBrush(QBrush(grad))
        p.drawEllipse(QRect(int(x), int(y), int(thumb_r), int(thumb_r)))
        p.end()


class SliderCard(QFrame):
    def __init__(self, title: str, rng: Tuple[int, int], val: int, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(8, 6, 8, 8)
        self.lay.setSpacing(6)

        self.lab = QLabel(title)
        self.lab.setObjectName("Subtle")
        self.lab.setAlignment(align_hcenter())
        self.lay.addWidget(self.lab)

        self.slider = QSlider(orientation_vertical())
        self.slider.setRange(rng[0], rng[1])
        self.slider.setValue(val)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lay.addWidget(self.slider, 1)

        self.readout = QLabel(str(val))
        self.readout.setObjectName("Subtle")
        self.readout.setAlignment(align_hcenter())
        self.lay.addWidget(self.readout)

        self.apply_size(92, 250, 180, 1.0)

    def apply_size(self, w: int, h: int, slider_h: int, scale: float):
        w = int(max(54, w))
        h = int(max(150, h))
        slider_h = int(max(90, slider_h))
        self.setFixedSize(w, h)
        self.slider.setMinimumHeight(slider_h)

        m_l = int(max(4, round(8 * scale)))
        m_t = int(max(4, round(6 * scale)))
        m_r = int(max(4, round(8 * scale)))
        m_b = int(max(4, round(8 * scale)))
        sp = int(max(3, round(6 * scale)))
        self.lay.setContentsMargins(m_l, m_t, m_r, m_b)
        self.lay.setSpacing(sp)

        f = self.lab.font()
        f.setPointSizeF(max(7.0, 9.0 * scale))
        self.lab.setFont(f)
        fr = self.readout.font()
        fr.setPointSizeF(max(7.0, 9.0 * scale))
        self.readout.setFont(fr)

    def set_value_text(self, v: int):
        self.readout.setText(str(int(v)))


class FxDragButton(QToolButton):
    """
    A QToolButton that can be dragged onto the FxTimeline.
    """
    def mouseMoveEvent(self, e):
        # Start drag when left button is held and moved.
        if hasattr(e, "buttons"):
            btns = e.buttons()
            left = Qt.MouseButton.LeftButton if hasattr(Qt, "MouseButton") else Qt.LeftButton
            if not (btns & left):
                return super().mouseMoveEvent(e)

        drag = QDrag(self)
        md = QMimeData()
        md.setData("application/x-djlighting-fx", self.text().strip().encode("utf-8"))
        drag.setMimeData(md)
        try:
            drag.setHotSpot(QPoint(10, 10))
        except Exception:
            pass
        drag.exec()
