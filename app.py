# app.py
"""
Single-Panel Lighting + Audio Controller (DJ / Club UI)

This version is set up for **Option 1: QLC+ bridge via Art-Net** (Windows).

✅ Your Python app sends Art-Net to 127.0.0.1, universe 0
✅ QLC+ receives it on Art-Net input, then outputs to your USB DMX dongle

Install:
  python -m pip install PyQt6 numpy sounddevice soundfile
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

# -------- Optional DMX (fallback stub if missing) --------
try:
    from dmx_output import OutputEngine, OutputConfig
    HAS_DMX = True
except Exception:
    HAS_DMX = False

    @dataclass
    class OutputConfig:
        enabled: bool = False
        blackout: bool = False
        protocol: str = "artnet"      # "off" | "usbpro" | "artnet"
        com_port: str = "COM3"
        baudrate: int = 57600
        target_ip: str = "127.0.0.1"
        universe: int = 0
        start_address: int = 1
        fps: int = 30

    class OutputEngine:
        def __init__(self):
            self._cfg = OutputConfig()
            self._last = [0] * 512

        def apply_config(self, cfg: OutputConfig):
            self._cfg = cfg

        def set_channels_from_values_0_255(self, values: List[int]):
            pass

        def tick(self):
            pass

        def close(self):
            pass


# -------- PyQt import (PyQt6 preferred, fallback to PyQt5) --------
try:
    from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QRect, QPointF
    from PyQt6.QtGui import (
        QFont, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy, QButtonGroup,
        QSlider, QLineEdit, QMessageBox, QFileDialog, QAbstractButton, QColorDialog,
        QBoxLayout
    )
    QT6 = True
except Exception:
    from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QRect, QPointF
    from PyQt5.QtGui import (
        QFont, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush
    )
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy, QButtonGroup,
        QSlider, QLineEdit, QMessageBox, QFileDialog, QAbstractButton, QColorDialog,
        QBoxLayout
    )
    QT6 = False


def _box_dir_left_to_right():
    return QBoxLayout.Direction.LeftToRight if QT6 else QBoxLayout.LeftToRight


def _box_dir_top_to_bottom():
    return QBoxLayout.Direction.TopToBottom if QT6 else QBoxLayout.TopToBottom


# ----------------------------- Styling -----------------------------
DARK_QSS = """
* { color: #E6E6E6; font-family: Segoe UI, Arial; }
QMainWindow { background: #15171A; }
QWidget { background: #15171A; }

QFrame#Panel {
    background: #1C1F23;
    border: 1px solid #2E333A;
    border-radius: 12px;
}

QLabel#PanelTitle {
    background: transparent;
    color: #D6D9DE;
    font-weight: 700;
    padding: 2px 2px;
}

QLabel#Subtle { color: #A9B0BA; }

QToolButton, QPushButton {
    background: #23272D;
    border: 1px solid #323844;
    border-radius: 12px;
    padding: 10px 14px;
}
QToolButton:hover, QPushButton:hover { border-color: #3C4452; }
QToolButton:checked {
    background: #2D3642;
    border-color: #5B89B8;
}

QPushButton#Primary { border-color: #5B89B8; }
QPushButton#Danger  { border-color: #B85B5B; }

QLineEdit {
    background: #121417;
    border: 1px solid #323844;
    border-radius: 10px;
    padding: 8px 10px;
}

/* DJ Fader (vertical) */
QSlider::groove:vertical {
    background: #0E1013;
    border: 1px solid #2E333A;
    width: 12px;
    border-radius: 6px;
}
QSlider::sub-page:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #7FB0E3, stop:1 #3E6EA6);
    border-radius: 6px;
}
QSlider::add-page:vertical { background: #0E1013; border-radius: 6px; }
QSlider::handle:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3B424D, stop:1 #222830);
    border: 1px solid #3A4350;
    height: 32px;
    margin: 0 -10px;
    border-radius: 10px;
}

/* Player scrubber (horizontal) */
QSlider::groove:horizontal {
    background: #0E1013;
    border: 1px solid #2E333A;
    height: 10px;
    border-radius: 5px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7FB0E3, stop:1 #3E6EA6);
    border-radius: 5px;
}
QSlider::add-page:horizontal { background: #0E1013; border-radius: 5px; }
QSlider::handle:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #E6EBF2, stop:1 #AAB2BC);
    border: 1px solid #3A4350;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

/* Color button */
QPushButton#ColorSwatch {
    border-radius: 12px;
    border: 1px solid #323844;
    padding: 12px;
    text-align: left;
}
QPushButton#ColorSwatch:hover { border-color: #3C4452; }
"""


# ----------------------------- Audio Analysis -----------------------------
def _band_mean(mag: np.ndarray, freqs: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.mean(mag[mask]))


@dataclass
class AudioFeatures:
    rms: float = 0.0
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    peak: float = 0.0
    bpm: float = 0.0
    spectrum: Optional[List[float]] = None


def compute_features_from_mono(mono: np.ndarray, sr: int, prev_rms: float) -> Tuple[AudioFeatures, float]:
    mono = mono.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    peak = max(0.0, rms - prev_rms)
    prev_rms = rms

    n = len(mono)
    if n < 64:
        return AudioFeatures(rms=rms, peak=peak), prev_rms

    win = np.hanning(n).astype(np.float32)
    y = mono * win
    spec = np.fft.rfft(y)
    mag = np.abs(spec).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    bass = _band_mean(mag, freqs, 20, 200)
    mid = _band_mean(mag, freqs, 200, 2000)
    treble = _band_mean(mag, freqs, 2000, 8000)

    nbands = 24
    fmin, fmax = 40.0, 8000.0
    edges = np.geomspace(fmin, fmax, nbands + 1).astype(np.float32)
    spectrum = [_band_mean(mag, freqs, float(edges[i]), float(edges[i + 1])) for i in range(nbands)]

    return AudioFeatures(rms=rms, bass=bass, mid=mid, treble=treble, peak=peak, spectrum=spectrum), prev_rms


# ----------------------------- UI Helpers -----------------------------
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


class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor if QT6 else Qt.PointingHandCursor)
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
        p.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing)

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

        grad = QLinearGradient(QPointF(x, y), QPointF(x, y + thumb_r))
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
        self.lab.setAlignment(Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter)
        self.lay.addWidget(self.lab)

        self.slider = QSlider(Qt.Orientation.Vertical if QT6 else Qt.Vertical)
        self.slider.setRange(rng[0], rng[1])
        self.slider.setValue(val)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lay.addWidget(self.slider, 1, Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter)

        self.readout = QLabel(str(val))
        self.readout.setObjectName("Subtle")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter)
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


# ----------------------------- Visualizer -----------------------------
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
        p.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing)

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

        p.setPen(Qt.PenStyle.NoPen if QT6 else Qt.NoPen)
        for i in range(nb):
            lvl = float(self._levels[i])
            bh = int(max_h * lvl)
            x = r.left() + i * (bar_w + gap)
            y = r.bottom() - bh
            p.fillRect(QRect(x, y, bar_w, bh), QColor(91, 137, 184, 200))

        p.end()


# ----------------------------- Audio WAV Worker -----------------------------
class AudioFileWorker(QObject):
    features = pyqtSignal(object)
    position = pyqtSignal(float, float)  # current_sec, total_sec
    error = pyqtSignal(str)
    running = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._stop_flag = False
        self._pause_flag = False
        self._stream = None

        self._audio: Optional[np.ndarray] = None
        self._sr = 48000
        self._idx = 0
        self._idx_lock = threading.Lock()
        self._total_frames = 0

        self._last_emit = 0.0
        self._emit_hz = 30.0
        self._prev_rms = 0.0

        self._onset_times: List[float] = []
        self._bpm_ema: float = 0.0
        self._bpm_alpha: float = 0.15

    def load_wav(self, path: str):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            if data.size == 0:
                raise ValueError("Empty file")
            self._audio = data
            self._sr = int(sr)
            with self._idx_lock:
                self._idx = 0
                self._total_frames = int(self._audio.shape[0])
            self._prev_rms = 0.0
            self._onset_times.clear()
            self._bpm_ema = 0.0
            self.position.emit(0.0, self.total_seconds())
        except Exception as e:
            self.error.emit(f"Failed to load WAV: {e}")

    def total_seconds(self) -> float:
        if self._sr <= 0 or self._total_frames <= 0:
            return 0.0
        return float(self._total_frames) / float(self._sr)

    def current_seconds(self) -> float:
        if self._sr <= 0:
            return 0.0
        with self._idx_lock:
            idx = int(self._idx)
        return float(idx) / float(self._sr)

    def seek_ratio(self, ratio: float):
        if self._audio is None:
            return
        ratio = float(np.clip(ratio, 0.0, 1.0))
        with self._idx_lock:
            self._idx = int(ratio * max(0, self._total_frames - 1))
        self._prev_rms = 0.0
        self._onset_times.clear()
        self._bpm_ema = 0.0
        self.position.emit(self.current_seconds(), self.total_seconds())

    def play(self):
        if self._audio is None:
            self.error.emit("No WAV loaded. Click Import first.")
            return

        if self._stream is not None:
            self._pause_flag = False
            self.running.emit(True)
            return

        self._stop_flag = False
        self._pause_flag = False
        channels = int(self._audio.shape[1])

        def callback(outdata, frames, t, status):
            if self._stop_flag:
                raise sd.CallbackStop()

            if self._pause_flag:
                outdata[:] = 0.0
                return

            with self._idx_lock:
                start = int(self._idx)
                end = start + frames
                chunk = self._audio[start:end]
                self._idx = end

            if len(chunk) < frames:
                outdata[:] = 0.0
                outdata[:len(chunk)] = chunk
                self._emit_features(chunk)
                self._stop_flag = True
                raise sd.CallbackStop()

            outdata[:] = chunk
            self._emit_features(chunk)

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=channels,
                dtype="float32",
                blocksize=1024,
                callback=callback,
            )
            self._stream.start()
            self.running.emit(True)
        except Exception as e:
            self._stream = None
            self.error.emit(f"Failed to start WAV playback: {e}")

    def stop(self):
        self._stop_flag = True
        self._pause_flag = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self.running.emit(False)
        self.position.emit(self.current_seconds(), self.total_seconds())

    def replay(self):
        self.stop()
        with self._idx_lock:
            self._idx = 0
        self._prev_rms = 0.0
        self._onset_times.clear()
        self._bpm_ema = 0.0
        self.position.emit(0.0, self.total_seconds())
        self.play()

    def _emit_features(self, chunk: np.ndarray):
        if chunk is None or len(chunk) < 64:
            self.position.emit(self.current_seconds(), self.total_seconds())
            return

        mono = chunk.mean(axis=1)
        feat, self._prev_rms = compute_features_from_mono(mono, self._sr, self._prev_rms)

        now_m = time.monotonic()
        if feat.peak > 0.012:
            self._onset_times.append(now_m)

        cutoff = now_m - 12.0
        while self._onset_times and self._onset_times[0] < cutoff:
            self._onset_times.pop(0)

        bpm = 0.0
        if len(self._onset_times) >= 4:
            intervals = np.diff(np.array(self._onset_times, dtype=np.float32))
            intervals = intervals[(intervals >= 0.272) & (intervals <= 1.5)]
            if len(intervals) >= 3:
                med = float(np.median(intervals))
                bpm = float(np.clip(60.0 / max(med, 1e-6), 40.0, 220.0))

        if bpm > 0.0:
            if self._bpm_ema <= 0.0:
                self._bpm_ema = bpm
            else:
                self._bpm_ema = (1.0 - self._bpm_alpha) * self._bpm_ema + self._bpm_alpha * bpm

        feat.bpm = float(self._bpm_ema)

        now = time.time()
        if now - self._last_emit >= (1.0 / self._emit_hz):
            self._last_emit = now
            self.features.emit(feat)
            self.position.emit(self.current_seconds(), self.total_seconds())


# ----------------------------- Lighting / FX -----------------------------
@dataclass
class ControlState:
    fx: str = ""
    auto_audio: bool = False
    auto_move: bool = False
    intensity: int = 80
    r: int = 0
    g: int = 120
    b: int = 255
    strobe_rate: int = 0
    pan: int = 270
    tilt: int = 90
    preset_name: str = "Untitled"


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    h = float(h % 1.0)
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


class FxEngine:
    def __init__(self):
        self._flash_until = 0.0
        self._lightning_until = 0.0
        self._strobe_phase = 0.0
        self._chase_phase = 0.0
        self._move_pan = 0.0
        self._move_tilt = 0.0
        self._move_pan_rate = 0.35
        self._move_tilt_rate = 0.28
        self._pan_bias = 0.0
        self._tilt_bias = 0.0

    def tick(self, state: ControlState, feat: AudioFeatures, dt: float, t: float):
        intensity = int(np.clip(state.intensity, 0, 100))
        r, g, b = int(state.r), int(state.g), int(state.b)

        pan_out = int(np.clip(state.pan, 0, 540))
        tilt_out = int(np.clip(state.tilt, 0, 180))

        if state.auto_audio:
            level = float(np.clip((feat.rms or 0.0) * 6.0, 0.0, 1.0))
            intensity = int(np.clip(0.25 * intensity + 0.75 * (level * 100.0), 0, 100))

            bass = float(feat.bass or 0.0)
            mid = float(feat.mid or 0.0)
            tre = float(feat.treble or 0.0)
            ssum = bass + mid + tre + 1e-6

            rr = tre / ssum
            gg = mid / ssum
            bb = bass / ssum

            r = int(np.clip(0.45 * r + 0.55 * (rr * 255.0), 0, 255))
            g = int(np.clip(0.45 * g + 0.55 * (gg * 255.0), 0, 255))
            b = int(np.clip(0.45 * b + 0.55 * (bb * 255.0), 0, 255))

        if state.auto_move:
            bpm = float(getattr(feat, "bpm", 0.0) or 0.0)
            speed = float(np.clip((bpm / 120.0) if bpm > 0.0 else 0.65, 0.35, 2.0))

            self._pan_bias += (np.sin(t * 0.11) + np.sin(t * 0.07 + 1.7)) * dt * 6.0
            self._tilt_bias += (np.sin(t * 0.09 + 0.4) + np.sin(t * 0.06 + 2.1)) * dt * 3.5
            self._pan_bias = float(np.clip(self._pan_bias, -60.0, 60.0))
            self._tilt_bias = float(np.clip(self._tilt_bias, -25.0, 25.0))

            self._move_pan += dt * self._move_pan_rate * speed
            self._move_tilt += dt * self._move_tilt_rate * speed

            amp_pan = 180.0
            amp_tilt = 55.0

            pan_center = float(state.pan)
            tilt_center = float(state.tilt)

            pan_out = int(np.clip(pan_center + np.sin(self._move_pan * 2.0 * np.pi) * amp_pan + self._pan_bias, 0, 540))
            tilt_out = int(np.clip(tilt_center + np.sin(self._move_tilt * 2.0 * np.pi + 0.8) * amp_tilt + self._tilt_bias, 0, 180))

        fx = (state.fx or "").strip().lower()
        strobe_on = True

        bpm = float(getattr(feat, "bpm", 0.0) or 0.0)
        beat_hz = (bpm / 60.0) if bpm > 0.0 else 1.6

        if fx == "blackout":
            intensity = 0
        elif fx == "rainbow":
            hue = (t * 0.08) % 1.0
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        elif fx == "wave":
            hue = (t * 0.06 + (pan_out / 540.0) * 0.2) % 1.0
            r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
            wave = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.5 * t))
            intensity = int(np.clip(intensity * wave, 0, 100))
        elif fx == "pulse":
            pulse = 0.30 + 0.70 * (0.5 + 0.5 * np.sin(2.0 * np.pi * beat_hz * t))
            intensity = int(np.clip(intensity * pulse, 0, 100))
        elif fx == "flash":
            if (feat.peak or 0.0) > 0.02 or (bpm > 0.0 and (np.sin(2.0 * np.pi * beat_hz * t) > 0.98)):
                self._flash_until = max(self._flash_until, t + 0.12)
            if t < self._flash_until:
                intensity = 100
                r, g, b = 255, 255, 255
        elif fx == "lightning":
            if t > self._lightning_until and np.random.rand() < (dt * 0.9):
                self._lightning_until = t + float(np.random.uniform(0.18, 0.55))
            if t < self._lightning_until:
                intensity = 100
                r, g, b = 255, 255, 255
                strobe_on = ((int(t * 26.0) % 2) == 0) and (np.random.rand() > 0.15)
        elif fx == "chase":
            self._chase_phase += dt * (beat_hz * 2.0)
            step = int(self._chase_phase) % 4
            chase_colors = [(255, 80, 60), (60, 255, 120), (80, 140, 255), (255, 255, 255)]
            r, g, b = chase_colors[step]
            intensity = int(np.clip(intensity * 0.95, 0, 100))
        elif fx == "strobe":
            rate = int(state.strobe_rate) if state.strobe_rate > 0 else 12
            self._strobe_phase += dt * float(rate)
            strobe_on = (int(self._strobe_phase) % 2) == 0

        if fx != "strobe" and state.strobe_rate > 0:
            self._strobe_phase += dt * float(state.strobe_rate)
            strobe_on = (int(self._strobe_phase) % 2) == 0

        if intensity > 0 and (r, g, b) == (0, 0, 0):
            r, g, b = 255, 255, 255

        if intensity <= 0:
            return 0, 0, 0, 0, False, pan_out, tilt_out

        return intensity, r, g, b, bool(strobe_on), pan_out, tilt_out


# ----------------------------- Simulation -----------------------------
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
        p.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing)

        R = self.rect().adjusted(14, 14, -14, -14)
        p.fillRect(R, QColor(10, 12, 16, 230))

        p.setPen(QPen(QColor(214, 217, 222, 180), 1))
        p.drawText(
            R.left() + 10,
            R.top() + 22,
            f"Simulation — FX: {self._fx_name}   |   Pan: {self._pan}°   Tilt: {self._tilt}°"
        )

        inner = R.adjusted(8, 32, -8, -10)
        fx_x = inner.center().x()
        fx_y = inner.bottom()

        pan_norm = (self._pan % 540) / 540.0
        tx = inner.left() + pan_norm * inner.width()

        tilt_norm = float(np.clip(self._tilt / 180.0, 0.0, 1.0))
        ty = inner.bottom() - tilt_norm * (inner.height() * 0.92)

        p.setPen(Qt.PenStyle.NoPen if QT6 else Qt.NoPen)
        p.setBrush(QColor(32, 36, 42, 255))
        p.drawEllipse(QRect(int(fx_x - 14), int(fx_y - 16), 28, 28))
        p.setBrush(QColor(70, 78, 90, 220))
        p.drawEllipse(QRect(int(fx_x - 7), int(fx_y - 9), 14, 14))

        if self._intensity <= 0 or not self._strobe_on:
            p.end()
            return

        r, g, b = self._rgb
        strength = self._intensity / 100.0
        a_beam = int(np.clip(200 * strength, 0, 220))
        a_hot = int(np.clip(220 * strength, 0, 240))

        dist = float(np.hypot(tx - fx_x, ty - fx_y))
        spread = float(np.clip(0.10 * dist + 25.0, 35.0, 220.0))
        half = spread * 0.5

        poly = QPolygonF([
            QPointF(fx_x - 10, fx_y - 6),
            QPointF(fx_x + 10, fx_y - 6),
            QPointF(tx + half, ty),
            QPointF(tx - half, ty),
        ])

        grad = QLinearGradient(QPointF(fx_x, fx_y), QPointF(tx, ty))
        grad.setColorAt(0.0, QColor(r, g, b, int(a_beam * 0.05)))
        grad.setColorAt(0.2, QColor(r, g, b, int(a_beam * 0.35)))
        grad.setColorAt(1.0, QColor(r, g, b, 0))

        p.setBrush(QBrush(grad))
        p.drawPolygon(poly)

        spot_r = float(np.clip(22.0 + 0.08 * spread, 28.0, 120.0))
        rg = QRadialGradient(QPointF(tx, ty), spot_r)
        rg.setColorAt(0.0, QColor(r, g, b, a_hot))
        rg.setColorAt(0.6, QColor(r, g, b, int(a_hot * 0.35)))
        rg.setColorAt(1.0, QColor(r, g, b, 0))
        p.setBrush(QBrush(rg))
        p.drawEllipse(QRect(int(tx - spot_r), int(ty - spot_r), int(spot_r * 2), int(spot_r * 2)))

        p.end()


def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    mm = int(sec // 60)
    ss = int(sec % 60)
    return f"{mm:02d}:{ss:02d}"


# ----------------------------- UI -----------------------------
class SinglePanel(QWidget):
    import_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    replay_clicked = pyqtSignal()
    seek_ratio = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = ControlState()

        self._user_scrubbing = False
        self._syncing_color = False

        self._base_w = 1366
        self._base_h = 768

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.split = QSplitter(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        root.addWidget(self.split, 1)

        # Left
        self.left = QWidget()
        self.left_lay = QVBoxLayout(self.left)
        self.left_lay.setContentsMargins(0, 0, 0, 0)
        self.left_lay.setSpacing(10)

        self.top_panel = Panel("Show Controls")
        self.left_lay.addWidget(self.top_panel, 0)

        status_row = QHBoxLayout()
        self.time_lbl = QLabel("00:00")
        self.time_lbl.setFont(QFont("Segoe UI", 14, 800))
        self.time_lbl.setObjectName("PanelTitle")
        self.status_lbl = QLabel("Stopped")
        self.status_lbl.setObjectName("Subtle")
        status_row.addWidget(self.time_lbl)
        status_row.addSpacing(10)
        status_row.addWidget(self.status_lbl)
        status_row.addStretch(1)
        self.top_panel.body.addLayout(status_row)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_name = QLineEdit(self.state.preset_name)
        self.preset_name.setPlaceholderText("Preset name (for SAVE/LOAD)")
        preset_row.addWidget(self.preset_name, 1)
        self.top_panel.body.addLayout(preset_row)

        rec_row = QHBoxLayout()
        rec_row.setSpacing(10)
        self.btn_record = QPushButton("RECORD")
        self.btn_stoprec = QPushButton("STOP")
        self.btn_stoprec.setObjectName("Danger")
        self.btn_save = QPushButton("SAVE")
        self.btn_load = QPushButton("LOAD")
        self._all_main_buttons = [self.btn_record, self.btn_stoprec, self.btn_save, self.btn_load]
        for b in self._all_main_buttons:
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        rec_row.addWidget(self.btn_record)
        rec_row.addWidget(self.btn_stoprec)
        rec_row.addWidget(self.btn_save)
        rec_row.addWidget(self.btn_load)
        self.top_panel.body.addLayout(rec_row)

        # FX
        self.fx_panel = Panel("Built-in FX (click again to turn off)")
        self.left_lay.addWidget(self.fx_panel, 0)

        self.fx_group = QButtonGroup(self)
        self.fx_group.setExclusive(False)
        self._active_fx_btn: Optional[QToolButton] = None

        self.fx_grid = QGridLayout()
        self.fx_grid.setSpacing(10)

        self.fx_buttons: Dict[str, QToolButton] = {}
        order = ["Lightning", "Rainbow", "Chase", "Blackout", "Wave", "Pulse", "Flash", "Strobe"]
        for i, name in enumerate(order):
            btn = QToolButton()
            btn.setText(name.upper())
            btn.setCheckable(True)
            btn.setMinimumHeight(56)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.fx_group.addButton(btn, i)
            self.fx_buttons[name] = btn
            self.fx_grid.addWidget(btn, i // 4, i % 4)

        self.fx_panel.body.addLayout(self.fx_grid)
        for btn in self.fx_buttons.values():
            btn.clicked.connect(lambda checked, b=btn: self._on_fx_clicked(b, checked))

        # Sliders
        self.slider_panel = Panel("Sliders")
        self.left_lay.addWidget(self.slider_panel, 1)

        self.row_and_picker = QBoxLayout(_box_dir_left_to_right())
        self.row_and_picker.setContentsMargins(0, 0, 0, 0)
        self.row_and_picker.setSpacing(12)

        self.row_box = QWidget()
        self.row = QHBoxLayout(self.row_box)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(8)

        self.card_dim = SliderCard("DIMMER", (0, 100), self.state.intensity)
        self.card_r = SliderCard("RED", (0, 255), self.state.r)
        self.card_g = SliderCard("GREEN", (0, 255), self.state.g)
        self.card_b = SliderCard("BLUE", (0, 255), self.state.b)
        self.card_st = SliderCard("STROBE", (0, 20), self.state.strobe_rate)
        self.card_pan = SliderCard("PAN", (0, 540), self.state.pan)
        self.card_tilt = SliderCard("TILT", (0, 180), self.state.tilt)

        self.cards = [self.card_dim, self.card_r, self.card_g, self.card_b, self.card_st, self.card_pan, self.card_tilt]
        for c in self.cards:
            self.row.addWidget(c)

        self.row_and_picker.addWidget(self.row_box, 0)

        self.color_panel = Panel("Color")
        self.color_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_swatch = QPushButton()
        self.color_swatch.setObjectName("ColorSwatch")
        self.color_swatch.setMinimumHeight(86)
        self.color_hex = QLabel("#000000")
        self.color_hex.setObjectName("Subtle")
        self.color_panel.body.addWidget(QLabel("Pick a color to set RGB sliders:"))
        self.color_panel.body.addWidget(self.color_swatch)
        self.color_panel.body.addWidget(self.color_hex)
        self.color_swatch.clicked.connect(self._pick_color_dialog)

        self.row_and_picker.addWidget(self.color_panel, 0)

        wrap = QWidget()
        wrap.setLayout(self.row_and_picker)
        self.slider_panel.body.addWidget(wrap)

        # Switches
        self.switch_row = QHBoxLayout()
        self.switch_row.setSpacing(16)

        def add_switch(text: str):
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            sw = ToggleSwitch()
            lb = QLabel(text)
            lb.setObjectName("Subtle")
            lay.addWidget(sw)
            lay.addWidget(lb)
            return sw, w, lb

        self.sw_auto, w_auto, self.lb_auto = add_switch("AUTO (Audio-driven)")
        self.sw_auto_move, w_move, self.lb_move = add_switch("AUTO PAN/TILT (BPM)")
        self.sw_dmx, w_dmx, self.lb_dmx = add_switch("DMX Output")

        self.switches = [self.sw_auto, self.sw_auto_move, self.sw_dmx]
        self.switch_labels = [self.lb_auto, self.lb_move, self.lb_dmx]

        self.switch_row.addWidget(w_auto)
        self.switch_row.addWidget(w_move)
        self.switch_row.addWidget(w_dmx)
        self.switch_row.addStretch(1)
        self.slider_panel.body.addLayout(self.switch_row)

        # Right
        self.right = QWidget()
        self.right_lay = QVBoxLayout(self.right)
        self.right_lay.setContentsMargins(0, 0, 0, 0)
        self.right_lay.setSpacing(10)

        self.sim_panel = Panel("Light Simulation")
        self.right_lay.addWidget(self.sim_panel, 1)
        self.sim = SimulationWidget()
        self.sim_panel.body.addWidget(self.sim, 1)

        self.audio_panel = Panel("Music / Audio")
        self.right_lay.addWidget(self.audio_panel, 0)

        bpm_row = QHBoxLayout()
        self.bpm_lbl = QLabel("BPM: --")
        self.bpm_lbl.setObjectName("Subtle")
        bpm_row.addWidget(self.bpm_lbl)
        bpm_row.addStretch(1)
        self.audio_panel.body.addLayout(bpm_row)

        self.visualizer = SimpleVisualizer(nbars=64)
        self.audio_panel.body.addWidget(self.visualizer)

        player_row = QHBoxLayout()
        player_row.setSpacing(10)
        self.time_cur = QLabel("00:00")
        self.time_cur.setObjectName("Subtle")
        self.time_tot = QLabel("00:00")
        self.time_tot.setObjectName("Subtle")

        self.scrub = QSlider(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.setValue(0)
        self.scrub.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.scrub.setFixedHeight(22)

        player_row.addWidget(self.time_cur)
        player_row.addWidget(self.scrub, 1)
        player_row.addWidget(self.time_tot)
        self.audio_panel.body.addLayout(player_row)

        self.scrub.sliderPressed.connect(self._scrub_start)
        self.scrub.sliderReleased.connect(self._scrub_commit)

        audio_row = QHBoxLayout()
        audio_row.setSpacing(10)
        self.btn_play = QPushButton("PLAY")
        self.btn_stop = QPushButton("STOP")
        self.btn_replay = QPushButton("REPLAY")
        self.btn_import = QPushButton("IMPORT")
        self.btn_play.setObjectName("Primary")
        self.btn_stop.setObjectName("Danger")
        self._transport_buttons = [self.btn_play, self.btn_stop, self.btn_replay, self.btn_import]

        audio_row.addWidget(self.btn_play)
        audio_row.addWidget(self.btn_stop)
        audio_row.addWidget(self.btn_replay)
        audio_row.addStretch(1)
        audio_row.addWidget(self.btn_import)
        self.audio_panel.body.addLayout(audio_row)

        self.wav_lbl = QLabel("WAV: (none)")
        self.wav_lbl.setObjectName("Subtle")
        self.wav_lbl.setWordWrap(True)
        self.audio_panel.body.addWidget(self.wav_lbl)

        self.split.addWidget(self.left)
        self.split.addWidget(self.right)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)

        # Wiring
        for s in [c.slider for c in self.cards]:
            s.valueChanged.connect(self._on_controls_changed)
        self.sw_auto.toggled.connect(self._on_controls_changed)
        self.sw_auto_move.toggled.connect(self._on_controls_changed)
        self.preset_name.textChanged.connect(self._on_controls_changed)

        self.btn_import.clicked.connect(self.import_clicked.emit)
        self.btn_play.clicked.connect(self.play_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_replay.clicked.connect(self.replay_clicked.emit)

        self._set_swatch_color(self.card_r.slider.value(), self.card_g.slider.value(), self.card_b.slider.value())
        self._on_controls_changed()
        QTimer.singleShot(0, self._apply_responsive)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_responsive()

    def _apply_responsive(self):
        w = max(1, self.width())
        h = max(1, self.height())
        sw = w / float(self._base_w)
        sh = h / float(self._base_h)
        s = float(np.clip(min(sw, sh), 0.58, 1.0))

        for p in [self.top_panel, self.fx_panel, self.slider_panel, self.sim_panel, self.audio_panel, self.color_panel]:
            p.apply_scale(s)

        btn_h = int(max(34, round(48 * s)))
        fx_h = int(max(30, round(56 * s)))
        for b in self._all_main_buttons:
            b.setMinimumHeight(btn_h)
        for b in self.fx_buttons.values():
            b.setMinimumHeight(fx_h)
        for b in self._transport_buttons:
            b.setMinimumHeight(int(max(30, round(44 * s))))

        f_time = QFont("Segoe UI", int(max(11, round(14 * s))), 800)
        self.time_lbl.setFont(f_time)
        for lb in [self.status_lbl, self.bpm_lbl, self.wav_lbl, self.time_cur, self.time_tot, self.color_hex] + self.switch_labels:
            f = lb.font()
            f.setPointSizeF(max(8.0, 10.0 * s))
            lb.setFont(f)

        for swt in self.switches:
            swt.apply_scale(s)

        self.visualizer.apply_scale(s)
        self.sim.apply_scale(s)
        self.scrub.setFixedHeight(int(max(18, round(22 * s))))
        self.color_swatch.setMinimumHeight(int(max(56, round(86 * s))))

        narrow = self.width() < 1120
        self.row_and_picker.setDirection(_box_dir_top_to_bottom() if narrow else _box_dir_left_to_right())

        if narrow:
            self.color_panel.setMaximumWidth(16777215)
        else:
            self.color_panel.setMaximumWidth(int(max(150, round(240 * s))))

        self._apply_slider_row_geometry(s, narrow)

    def _apply_slider_row_geometry(self, s: float, narrow: bool):
        n = len(self.cards)
        spacing = int(max(4, round(8 * s)))
        self.row.setSpacing(spacing)
        self.row_and_picker.setSpacing(int(max(6, round(12 * s))))

        avail = self.row_box.width()
        if avail <= 10:
            avail = max(10, self.slider_panel.width() - 60)

        if not narrow:
            avail -= (self.color_panel.width() + int(max(6, round(12 * s))))

        avail = max(10, avail)

        card_w = int((avail - spacing * (n - 1)) / n)
        card_w = int(np.clip(card_w, 54, 110))

        card_h = int(max(150, round(250 * s)))
        slider_h = int(max(90, round(180 * s)))

        if self.height() < 560:
            card_h = int(max(140, card_h - 20))
            slider_h = int(max(80, slider_h - 15))

        for c in self.cards:
            c.apply_size(card_w, card_h, slider_h, s)

    def _scrub_start(self):
        self._user_scrubbing = True

    def _scrub_commit(self):
        self._user_scrubbing = False
        ratio = float(self.scrub.value()) / 1000.0
        self.seek_ratio.emit(ratio)

    def set_player_position(self, cur_sec: float, total_sec: float):
        self.time_cur.setText(_fmt_time(cur_sec))
        self.time_tot.setText(_fmt_time(total_sec))
        if not self._user_scrubbing and total_sec > 0.0:
            ratio = float(np.clip(cur_sec / total_sec, 0.0, 1.0))
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(ratio * 1000))
            self.scrub.blockSignals(False)

    def _set_swatch_color(self, r: int, g: int, b: int):
        r = int(np.clip(r, 0, 255))
        g = int(np.clip(g, 0, 255))
        b = int(np.clip(b, 0, 255))
        hexv = f"#{r:02X}{g:02X}{b:02X}"
        self.color_hex.setText(hexv)
        self.color_swatch.setText(f" {hexv}")
        self.color_swatch.setStyleSheet(
            f"QPushButton#ColorSwatch{{"
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 rgba({r},{g},{b},220), stop:1 rgba({r},{g},{b},140));"
            f"}}"
        )

    def _pick_color_dialog(self):
        current = QColor(int(self.card_r.slider.value()), int(self.card_g.slider.value()), int(self.card_b.slider.value()))
        c = QColorDialog.getColor(current, self, "Pick Color")
        if not c.isValid():
            return
        self._syncing_color = True
        try:
            self.card_r.slider.setValue(c.red())
            self.card_g.slider.setValue(c.green())
            self.card_b.slider.setValue(c.blue())
        finally:
            self._syncing_color = False
        self._set_swatch_color(self.card_r.slider.value(), self.card_g.slider.value(), self.card_b.slider.value())
        self._on_controls_changed()

    def _on_fx_clicked(self, btn: QToolButton, checked: bool):
        if checked:
            if self._active_fx_btn is not None and self._active_fx_btn is not btn:
                self._active_fx_btn.blockSignals(True)
                self._active_fx_btn.setChecked(False)
                self._active_fx_btn.blockSignals(False)
            self._active_fx_btn = btn
            self.state.fx = btn.text().strip().title()
            self.status_lbl.setText(f"FX: {self.state.fx}")
        else:
            if self._active_fx_btn is btn:
                self._active_fx_btn = None
            self.state.fx = ""
            self.status_lbl.setText("FX: Off")

    def _on_controls_changed(self, *_):
        self.state.preset_name = self.preset_name.text().strip() or "Untitled"
        self.state.intensity = int(self.card_dim.slider.value())
        self.state.r = int(self.card_r.slider.value())
        self.state.g = int(self.card_g.slider.value())
        self.state.b = int(self.card_b.slider.value())
        self.state.strobe_rate = int(self.card_st.slider.value())
        self.state.pan = int(self.card_pan.slider.value())
        self.state.tilt = int(self.card_tilt.slider.value())
        self.state.auto_audio = bool(self.sw_auto.isChecked())
        self.state.auto_move = bool(self.sw_auto_move.isChecked())

        self.card_dim.set_value_text(self.state.intensity)
        self.card_r.set_value_text(self.state.r)
        self.card_g.set_value_text(self.state.g)
        self.card_b.set_value_text(self.state.b)
        self.card_st.set_value_text(self.state.strobe_rate)
        self.card_pan.set_value_text(self.state.pan)
        self.card_tilt.set_value_text(self.state.tilt)

        lock = bool(self.state.auto_move)
        self.card_pan.slider.setEnabled(not lock)
        self.card_tilt.slider.setEnabled(not lock)

        if not self._syncing_color:
            self._set_swatch_color(self.state.r, self.state.g, self.state.b)

    def set_pan_tilt_from_engine(self, pan: int, tilt: int):
        if not self.state.auto_move:
            return
        pan = int(np.clip(pan, 0, 540))
        tilt = int(np.clip(tilt, 0, 180))

        self.card_pan.slider.blockSignals(True)
        self.card_tilt.slider.blockSignals(True)
        self.card_pan.slider.setValue(pan)
        self.card_tilt.slider.setValue(tilt)
        self.card_pan.slider.blockSignals(False)
        self.card_tilt.slider.blockSignals(False)

        self.card_pan.set_value_text(pan)
        self.card_tilt.set_value_text(tilt)

        self.state.pan = pan
        self.state.tilt = tilt

    def set_loaded_file(self, path: str):
        self.wav_lbl.setText(f"WAV: {path}")

    def update_visualizer(self, feat: AudioFeatures):
        self.visualizer.update_from_spectrum(feat.spectrum)
        self.bpm_lbl.setText(f"BPM: {feat.bpm:0.0f}" if (feat.bpm and feat.bpm > 0.0) else "BPM: --")


# ----------------------------- Main Window -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lighting FX + Audio Visualizer + Simulation (Single Panel)")
        self.setMinimumSize(854, 480)

        self.ui = SinglePanel()
        self.setCentralWidget(self.ui)

        # ---------- OUTPUT ENGINE ----------
        self.output = OutputEngine()

        # ✅ OPTION 1: Art-Net -> QLC+ -> USB DMX
        # QLC+ screenshot shows "ArtNet Universe = 0" for Universe 1.
        self.output_cfg = OutputConfig(
            enabled=False,
            protocol="artnet",
            target_ip="127.0.0.1",
            universe=0,          # MUST match QLC+ ArtNet Universe
            start_address=1,     # your fixture DMX address is 001
            fps=30,
        )
        self.output.apply_config(self.output_cfg)

        self.output_timer = QTimer(self)
        self.output_timer.setInterval(10)
        self.output_timer.timeout.connect(self.output.tick)
        self.output_timer.start()

        # ---------- AUDIO THREAD ----------
        self.file_thread = QThread(self)
        self.file_worker = AudioFileWorker()
        self.file_worker.moveToThread(self.file_thread)
        self.file_thread.start()

        self.file_worker.features.connect(self._on_audio_features)
        self.file_worker.position.connect(self._on_audio_position)
        self.file_worker.error.connect(self._on_audio_error)

        self.ui.import_clicked.connect(self._import_wav)
        self.ui.play_clicked.connect(self._play)
        self.ui.stop_clicked.connect(self._stop)
        self.ui.replay_clicked.connect(self._replay)
        self.ui.seek_ratio.connect(self._seek_ratio)

        self.fx = FxEngine()
        self._latest_feat = AudioFeatures()
        self._last_t = time.monotonic()

        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(33)
        self.frame_timer.timeout.connect(self._frame_tick)
        self.frame_timer.start()

        self._wav_path: Optional[str] = None

    def closeEvent(self, event):
        try:
            QTimer.singleShot(0, self.file_worker.stop)
        except Exception:
            pass
        try:
            self.output.close()
        except Exception:
            pass
        try:
            self.file_thread.quit()
            self.file_thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)

    def _import_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import WAV", os.getcwd(), "WAV Files (*.wav)")
        if not path:
            return
        self._wav_path = path
        self.ui.set_loaded_file(path)
        QTimer.singleShot(0, lambda: self.file_worker.load_wav(path))

    def _play(self):
        if not self._wav_path:
            QMessageBox.information(self, "No WAV", "Click IMPORT first.")
            return
        QTimer.singleShot(0, self.file_worker.play)

    def _stop(self):
        QTimer.singleShot(0, self.file_worker.stop)

    def _replay(self):
        if not self._wav_path:
            QMessageBox.information(self, "No WAV", "Click IMPORT first.")
            return
        QTimer.singleShot(0, self.file_worker.replay)

    def _seek_ratio(self, ratio: float):
        QTimer.singleShot(0, lambda r=ratio: self.file_worker.seek_ratio(r))

    def _on_audio_features(self, feat_obj):
        feat: AudioFeatures = feat_obj
        self._latest_feat = feat
        self.ui.update_visualizer(feat)

    def _on_audio_position(self, cur_sec: float, total_sec: float):
        self.ui.set_player_position(cur_sec, total_sec)

    def _on_audio_error(self, msg: str):
        QMessageBox.critical(self, "Audio Error", msg)

    def _frame_tick(self):
        now = time.monotonic()
        dt = max(0.001, now - self._last_t)
        self._last_t = now

        intensity, r, g, b, st_on, pan_out, tilt_out = self.fx.tick(self.ui.state, self._latest_feat, dt, now)
        self.ui.set_pan_tilt_from_engine(pan_out, tilt_out)

        fx_name = self.ui.state.fx if self.ui.state.fx else "Off"
        self.ui.sim.set_state(intensity, r, g, b, st_on, fx_name, pan_out, tilt_out)

        # ---------------- DMX OUTPUT (DJScorpio 14CH - manual-matched) ----------------
        if HAS_DMX:
            try:
                self.output_cfg.enabled = bool(self.ui.sw_dmx.isChecked())
                self.output_cfg.blackout = (self.ui.state.fx.strip().lower() == "blackout")
                self.output.apply_config(self.output_cfg)

                fx = (self.ui.state.fx or "Off").strip().upper()
                bpm = float(getattr(self._latest_feat, "bpm", 0.0) or 0.0)

                def clamp01(x: float) -> float:
                    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

                dimmer_255 = int(max(0, min(255, (float(intensity) / 100.0) * 255.0)))

                pan_255 = int(max(0, min(255, (float(pan_out) / 540.0) * 255.0)))
                tilt_255 = int(max(0, min(255, (float(tilt_out) / 180.0) * 255.0)))
                pan_fine = 0
                tilt_fine = 0

                motor_speed = 0

                rate = int(getattr(self.ui.state, "strobe_rate", 0))
                if (not st_on) and fx not in ("STROBE", "FLASH", "LIGHTNING"):
                    flash = 0
                else:
                    if rate <= 0:
                        flash = 32
                    else:
                        flash = int(16 + (min(rate, 20) / 20.0) * (255 - 16))

                def wheel_value_from_rgb(rr: int, gg: int, bb: int) -> int:
                    wheel = [
                        (5,  (255, 255, 255)),  # white
                        (15, (255, 0, 0)),      # red
                        (25, (0, 255, 0)),      # green
                        (35, (0, 0, 255)),      # blue
                        (45, (255, 255, 0)),    # yellow
                        (55, (255, 140, 0)),    # orange
                        (65, (160, 0, 255)),    # purple
                        (75, (120, 70, 20)),    # brown
                        (85, (255, 0, 90)),     # rose red
                        (95, (255, 180, 120)),  # warming
                        (105, (120, 255, 120)), # light green
                        (115, (120, 180, 255)), # light blue
                    ]
                    best_v = 5
                    best_d = 10**18
                    for v, (wr, wg, wb) in wheel:
                        d = (rr - wr) ** 2 + (gg - wg) ** 2 + (bb - wb) ** 2
                        if d < best_d:
                            best_d = d
                            best_v = v
                    return best_v

                if fx == "RAINBOW":
                    t01 = clamp01(bpm / 200.0) if bpm > 0 else 0.5
                    color = int(187 - t01 * (187 - 120))  # forward
                elif fx == "WAVE":
                    t01 = clamp01(bpm / 200.0) if bpm > 0 else 0.5
                    color = int(188 + t01 * (255 - 188))  # reverse
                else:
                    color = wheel_value_from_rgb(int(r), int(g), int(b))

                if fx == "CHASE":
                    t01 = clamp01(bpm / 200.0) if bpm > 0 else 0.5
                    pattern = int(217 - t01 * (217 - 180))
                elif fx == "WAVE":
                    t01 = clamp01(bpm / 200.0) if bpm > 0 else 0.5
                    pattern = int(218 + t01 * (255 - 218))
                elif fx == "PULSE":
                    pattern = 92
                elif fx == "LIGHTNING":
                    pattern = 2
                else:
                    pattern = 0

                if fx in ("FLASH", "STROBE"):
                    t01 = clamp01(bpm / 200.0) if bpm > 0 else 0.5
                    prism = int(128 + t01 * (255 - 128))
                else:
                    prism = 0

                if bool(self.ui.sw_auto.isChecked()):
                    mode = 220
                elif fx in ("CHASE", "RAINBOW", "WAVE"):
                    mode = 80
                else:
                    mode = 0

                reset = 0

                strip_fx = 0 if fx in ("OFF", "BLACKOUT") else 160
                if bpm > 0:
                    t01 = clamp01(bpm / 200.0)
                    strip_speed = int(255 - t01 * 255)  # 0 fast -> 255 slow
                else:
                    strip_speed = 120

                values = [
                    pan_255,      # CH1
                    pan_fine,     # CH2
                    tilt_255,     # CH3
                    tilt_fine,    # CH4
                    motor_speed,  # CH5
                    dimmer_255,   # CH6
                    flash,        # CH7
                    color,        # CH8
                    pattern,      # CH9
                    prism,        # CH10
                    mode,         # CH11
                    reset,        # CH12
                    strip_fx,     # CH13
                    strip_speed,  # CH14
                ]
                self.output.set_channels_from_values_0_255(values)
            except Exception:
                pass


def main():
    app = QApplication([])
    app.setStyleSheet(DARK_QSS)
    w = MainWindow()
    w.showMaximized()
    if QT6:
        app.exec()
    else:
        app.exec_()


if __name__ == "__main__":
    main()
