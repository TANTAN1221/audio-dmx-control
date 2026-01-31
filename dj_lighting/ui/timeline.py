
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..qt_compat import (
    QWidget, Qt, QPainter, QPen, QColor, QBrush, QFont, QRectF,
    QMimeData, QPoint, pyqtSignal, QMenu, QAction
)


MIME_FX = "application/x-djlighting-fx"


@dataclass
class FxCue:
    t: float          # seconds
    fx: str           # effect name
    dur: float = 0.6  # seconds (how long to apply)


class FxTimeline(QWidget):
    """
    A lightweight drop-target timeline that holds FX cues aligned to audio time.

    - Drag an FX button onto the timeline to create a cue
    - Right-click a cue to delete it
    - Emits cuesChanged whenever cues are edited
    """
    cuesChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(86)
        self.setObjectName("FxTimeline")

        self._cur = 0.0
        self._total = 0.0
        self._cues: List[FxCue] = []
        self._beats: List[float] = []
        self._bpm: float = 0.0
        self._hover_idx: Optional[int] = None

        self._font = QFont("Segoe UI", 9)
        self._font_bold = QFont("Segoe UI", 9, 700)

    # ----- public API -----
    def set_position(self, cur_sec: float, total_sec: float):
        self._cur = max(0.0, float(cur_sec))
        self._total = max(0.0, float(total_sec))
        self.update()

    def cues(self) -> List[FxCue]:
        return list(self._cues)


    def set_beats(self, bpm: float, beat_times: List[float]):
        self._bpm = float(bpm or 0.0)
        self._beats = [float(t) for t in (beat_times or []) if t >= 0.0]
        self.update()

    def set_cues(self, cues: List[dict] | List[FxCue]):
        out: List[FxCue] = []
        for c in cues:
            if isinstance(c, FxCue):
                out.append(c)
            else:
                out.append(FxCue(float(c.get("t", 0.0)), str(c.get("fx", "")), float(c.get("dur", 0.6))))
        out.sort(key=lambda x: x.t)
        self._cues = out
        self.cuesChanged.emit()
        self.update()

    def add_cue(self, cue: FxCue):
        self._cues.append(cue)
        self._cues.sort(key=lambda x: x.t)
        self.cuesChanged.emit()
        self.update()

    def export_cues(self) -> List[dict]:
        return [{"t": float(c.t), "fx": c.fx, "dur": float(c.dur)} for c in self._cues]

    def take_triggers(self, prev_sec: float, cur_sec: float) -> List[FxCue]:
        """
        Return cues whose timestamp is in (prev_sec, cur_sec] (forward playback).
        """
        if cur_sec <= prev_sec:
            return []
        hits = [c for c in self._cues if prev_sec < c.t <= cur_sec]
        return hits

    # ----- helpers -----
    def _timeline_rect(self) -> QRectF:
        pad = 12
        return QRectF(pad, 28, max(1.0, self.width() - 2 * pad), max(1.0, self.height() - 40))

    def _x_to_time(self, x: float) -> float:
        if self._total <= 0:
            return 0.0
        r = self._timeline_rect()
        u = (x - r.left()) / max(1.0, r.width())
        u = min(1.0, max(0.0, u))
        return u * self._total

    def _time_to_x(self, t: float) -> float:
        r = self._timeline_rect()
        if self._total <= 0:
            return r.left()
        u = min(1.0, max(0.0, t / self._total))
        return r.left() + u * r.width()

    def _cue_at(self, pos) -> Optional[int]:
        r = self._timeline_rect()
        y0 = r.top()
        y1 = r.bottom()
        for i, c in enumerate(self._cues):
            cx = self._time_to_x(c.t)
            hit = QRectF(cx - 8, y0, 16, y1 - y0)
            if hit.contains(pos):
                return i
        return None

    # ----- painting -----
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # background
        p.fillRect(self.rect(), QColor(0, 0, 0, 0))

        r = self._timeline_rect()

        # title / hint
        p.setFont(self._font_bold)
        p.setPen(QColor(220, 220, 220))
        p.drawText(12, 18, "Audio timeline — drag FX here to create cues")

        # base rail
        p.setPen(QPen(QColor(80, 85, 95), 2))
        p.drawRoundedRect(r, 10, 10)

        # ticks
        p.setFont(self._font)
        p.setPen(QColor(150, 155, 165))
        if self._total > 0:
            # choose ~8-12 major ticks
            n = 10
            for i in range(n + 1):
                x = r.left() + (i / n) * r.width()
                p.drawLine(int(x), int(r.top()), int(x), int(r.top()) - 6)
                t = (i / n) * self._total
                mm = int(t // 60)
                ss = int(t % 60)
                p.drawText(int(x - 12), int(r.top()) - 10, f"{mm:02d}:{ss:02d}")

        
        # beat grid (if available)
        if self._total > 0 and self._beats:
            # faint beat lines + slightly stronger bar lines (every 4 beats)
            p.setPen(QPen(QColor(120, 125, 135, 90), 1))
            for bi, bt in enumerate(self._beats):
                if bt < 0 or bt > self._total:
                    continue
                x = self._time_to_x(bt)
                is_bar = (bi % 4 == 0)
                if is_bar:
                    p.setPen(QPen(QColor(150, 155, 170, 140), 1))
                else:
                    p.setPen(QPen(QColor(120, 125, 135, 90), 1))
                p.drawLine(int(x), int(r.top()+2), int(x), int(r.bottom()-2))

        # cues
        for i, c in enumerate(self._cues):
            cx = self._time_to_x(c.t)
            cue_rect = QRectF(cx - 7, r.top() + 8, 14, r.height() - 16)
            col = QColor(120, 190, 255) if i != self._hover_idx else QColor(170, 220, 255)
            p.setBrush(QBrush(col))
            p.setPen(QPen(QColor(0, 0, 0, 0), 0))
            p.drawRoundedRect(cue_rect, 6, 6)

            # small label
            p.setPen(QColor(220, 220, 220))
            p.setFont(self._font)
            label = c.fx.upper()
            p.drawText(int(cx + 10), int(r.top() + 22), label)

        # playhead
        if self._total > 0:
            x = self._time_to_x(self._cur)
            p.setPen(QPen(QColor(240, 240, 240), 2))
            p.drawLine(int(x), int(r.top()), int(x), int(r.bottom()))

    # ----- DnD -----
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_FX):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if not e.mimeData().hasFormat(MIME_FX):
            e.ignore()
            return
        self._hover_idx = self._cue_at(e.position() if hasattr(e, "position") else e.pos())
        e.acceptProposedAction()
        self.update()

    def dragLeaveEvent(self, e):
        self._hover_idx = None
        self.update()

    def dropEvent(self, e):
        if not e.mimeData().hasFormat(MIME_FX):
            e.ignore()
            return
        fx_name = bytes(e.mimeData().data(MIME_FX)).decode("utf-8", errors="ignore").strip()
        pos = e.position() if hasattr(e, "position") else e.pos()
        t = self._x_to_time(pos.x())
        if fx_name:
            self.add_cue(FxCue(t=t, fx=fx_name, dur=0.6))
        e.acceptProposedAction()

    # ----- context menu -----
    def contextMenuEvent(self, e):
        idx = self._cue_at(e.pos())
        if idx is None:
            return
        menu = QMenu(self)
        act_del = QAction("Delete cue", self)
        menu.addAction(act_del)

        def do_del():
            if 0 <= idx < len(self._cues):
                self._cues.pop(idx)
                self.cuesChanged.emit()
                self.update()

        act_del.triggered.connect(do_del)
        menu.exec(e.globalPos())
