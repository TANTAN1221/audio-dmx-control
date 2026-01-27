"""
Single-Panel Lighting + Audio Controller (Clean UI)
- Built-in FX buttons (toggle-off by clicking again): wave, pulse, flash, strobe, lightning, rainbow, chase, blackout
- Sliders: dimmer (intensity), RGB, strobe rate, PAN, TILT
- Toggles:
    - AUTO (Audio-driven): mixes FX + your chosen RGB, audio-reactive intensity
    - AUTO PAN/TILT (BPM): moving-head movement from BPM (locks PAN/TILT sliders)
    - DMX Output
- Presets: record, stop, save, load (JSON snapshot of current UI state)
- Audio: import WAV, play, stop, replay + simple visualizer
- Big right panel: light simulation (works even without hardware)
- Optional DMX Output (Art-Net / sACN) via dmx_output.py if present

Preset/timecode behavior:
- Timecode does NOT run on startup.
- Clicking RECORD starts timecode and marks "recording".
- Clicking STOP ends recording; ONLY then you can SAVE.
- SAVE is blocked unless you have stopped after recording.

Install:
  python -m pip install PyQt6 numpy sounddevice soundfile

Optional (DMX):
  Put dmx_output.py in the same folder as this file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
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
        protocol: str = "off"
        target_ip: str = "255.255.255.255"
        universe: int = 0
        sacn_universe: int = 1
        start_address: int = 1
        fps: int = 30

    class OutputEngine:
        def __init__(self):
            self._cfg = OutputConfig()
            self._last = [0] * 512

        def apply_config(self, cfg: OutputConfig):
            self._cfg = cfg

        def set_channels_from_faders(self, faders_0_100: List[int]):
            for i, v in enumerate(faders_0_100):
                if 0 <= i < len(self._last):
                    self._last[i] = int(max(0, min(100, v)))

        def tick(self):
            pass

        def close(self):
            pass


# -------- PyQt import (PyQt6 preferred, fallback to PyQt5) --------
try:
    from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QRect, QPointF, QSize
    from PyQt6.QtGui import (
        QFont, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy, QButtonGroup,
        QSlider, QLineEdit, QMessageBox, QFileDialog, QAbstractButton
    )
    QT6 = True
except Exception:
    from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QRect, QPointF, QSize
    from PyQt5.QtGui import (
        QFont, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush
    )
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy, QButtonGroup,
        QSlider, QLineEdit, QMessageBox, QFileDialog, QAbstractButton
    )
    QT6 = False


# ----------------------------- Styling -----------------------------
DARK_QSS = """
* { color: #E6E6E6; }
QMainWindow { background: #15171A; }
QWidget { background: #15171A; }

QFrame#Panel {
    background: #1C1F23;
    border: 1px solid #2E333A;
    border-radius: 10px;
}

QLabel#PanelTitle {
    background: transparent;
    color: #D6D9DE;
    font-weight: 600;
    padding: 2px 2px;
}

QLabel#Subtle { color: #A9B0BA; }

QToolButton, QPushButton {
    background: #23272D;
    border: 1px solid #323844;
    border-radius: 999px;
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

QFrame#SliderWell {
    background: #171A1E;
    border: 1px solid #2A3038;
    border-radius: 10px;
}

QSlider { background: transparent; }

QSlider::groove:vertical {
    width: 14px;
    border-radius: 7px;
    border: 1px solid #2A2F36;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0E1013, stop:0.45 #1B1F24, stop:0.55 #1B1F24, stop:1 #0E1013);
}

QSlider::sub-page:vertical {
    width: 14px;
    border-radius: 7px;
    background: qlineargradient(x1:0, y1:1, x2:0, y2:0,
        stop:0 #2C3846, stop:1 #5B89B8);
}

QSlider::add-page:vertical {
    width: 14px;
    border-radius: 7px;
    background: transparent;
}

QSlider::handle:vertical {
    width: 28px;
    height: 36px;
    margin: 0 -8px;
    border-radius: 6px;
    border: 1px solid #3B4450;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3D434B, stop:0.45 #22272E, stop:0.55 #1A1E24, stop:1 #353B43);
}

QSlider::handle:vertical:hover {
    border: 1px solid #5B89B8;
}
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        if title:
            t = QLabel(title)
            t.setObjectName("PanelTitle")
            outer.addWidget(t)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(10)
        outer.addLayout(self.body)


class ToggleSwitch(QAbstractButton):
    toggled = pyqtSignal(bool)

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self._text = text
        self.setCursor(Qt.CursorShape.PointingHandCursor if QT6 else Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        self._text = text
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        w = 54 + 10 + max(40, int(len(self._text) * 7.5))
        h = 28
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.toggled.emit(self.isChecked())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing)

        R = self.rect()
        switch_w = 46
        switch_h = 22
        x0 = R.left()
        y0 = R.center().y() - switch_h // 2
        track = QRect(x0, y0, switch_w, switch_h)

        checked = self.isChecked()

        grad = QLinearGradient(QPointF(track.left(), track.top()), QPointF(track.left(), track.bottom()))
        if checked:
            grad.setColorAt(0.0, QColor(55, 85, 120, 230))
            grad.setColorAt(1.0, QColor(35, 55, 80, 230))
        else:
            grad.setColorAt(0.0, QColor(22, 25, 30, 240))
            grad.setColorAt(1.0, QColor(12, 14, 17, 240))

        p.setPen(QPen(QColor(60, 70, 85, 160), 1))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(track, 11, 11)

        knob_r = 9
        pad = 3
        if checked:
            kx = track.right() - pad - knob_r
        else:
            kx = track.left() + pad + knob_r
        ky = track.center().y()

        kgrad = QRadialGradient(QPointF(kx - 3, ky - 3), knob_r * 1.8)
        kgrad.setColorAt(0.0, QColor(235, 238, 242, 255))
        kgrad.setColorAt(1.0, QColor(120, 130, 145, 255))
        p.setPen(QPen(QColor(20, 22, 26, 180), 1))
        p.setBrush(QBrush(kgrad))
        p.drawEllipse(QPointF(kx, ky), knob_r, knob_r)

        p.setPen(QPen(QColor(230, 232, 235, 220), 1))
        tx = track.right() + 10
        p.drawText(
            QRect(tx, R.top(), R.right() - tx, R.height()),
            Qt.AlignmentFlag.AlignVCenter if QT6 else Qt.AlignVCenter,
            self._text,
        )

        p.end()


class SliderWell(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SliderWell")


# ----------------------------- Simple Visualizer -----------------------------
class SimpleVisualizer(QFrame):
    def __init__(self, nbars: int = 64, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.nbars = nbars
        self._levels = np.zeros(nbars, dtype=np.float32)
        self._ema = np.ones(nbars, dtype=np.float32) * 1e-3
        self._alpha = 0.08
        self.setMinimumHeight(110)

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

        grid_pen = QPen(QColor(46, 51, 58, 160))
        p.setPen(grid_pen)
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


# ----------------------------- Audio WAV Worker (with BPM estimation) -----------------------------
class AudioFileWorker(QObject):
    features = pyqtSignal(object)   # AudioFeatures
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
            self._idx = 0
            self._prev_rms = 0.0
            self._onset_times.clear()
            self._bpm_ema = 0.0
        except Exception as e:
            self.error.emit(f"Failed to load WAV: {e}")

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

            end = self._idx + frames
            chunk = self._audio[self._idx:end]

            if len(chunk) < frames:
                outdata[:] = 0.0
                outdata[:len(chunk)] = chunk
                self._idx = end
                self._emit_features(chunk)
                self._stop_flag = True
                raise sd.CallbackStop()

            outdata[:] = chunk
            self._idx = end
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

    def replay(self):
        self.stop()
        self._idx = 0
        self._prev_rms = 0.0
        self._onset_times.clear()
        self._bpm_ema = 0.0
        self.play()

    def _emit_features(self, chunk: np.ndarray):
        if chunk is None or len(chunk) < 64:
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
            intervals = intervals[(intervals >= 0.272) & (intervals <= 1.5)]  # 220..40 bpm
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


# ----------------------------- Lighting / FX -----------------------------
FX_LIST = ["Wave", "Pulse", "Flash", "Strobe", "Lightning", "Rainbow", "Chase", "Blackout"]


@dataclass
class ControlState:
    fx: str = ""
    auto_audio: bool = False
    auto_move: bool = False      # NEW: auto PAN/TILT from BPM (locks sliders)
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

        self._move_phase_pan = 0.0
        self._move_phase_tilt = 0.0

    @staticmethod
    def _blend_rgb(base: Tuple[int, int, int], fx: Tuple[int, int, int], mix: float) -> Tuple[int, int, int]:
        mix = float(np.clip(mix, 0.0, 1.0))
        br, bg, bb = base
        fr, fg, fb = fx
        r = int(np.clip(br * (1.0 - mix) + fr * mix, 0, 255))
        g = int(np.clip(bg * (1.0 - mix) + fg * mix, 0, 255))
        b = int(np.clip(bb * (1.0 - mix) + fb * mix, 0, 255))
        return r, g, b

    def tick(
        self,
        state: ControlState,
        feat: AudioFeatures,
        dt: float,
        t: float
    ) -> Tuple[int, int, int, int, bool, int, int]:
        intensity = int(max(0, min(100, state.intensity)))
        base_rgb = (int(state.r), int(state.g), int(state.b))

        pan_out = int(np.clip(state.pan, 0, 540))
        tilt_out = int(np.clip(state.tilt, 0, 180))

        auto_mix = 0.0

        # Audio-driven intensity + color mix (kept separate from movement)
        if state.auto_audio and feat is not None:
            a = float(np.clip(feat.rms * 10.0, 0.0, 1.0))
            intensity = int(np.clip(20 + a * 80, 0, 100))
            auto_mix = float(np.clip(0.25 + a * 0.55, 0.25, 0.80))

            if feat.peak > 0.010:
                self._flash_until = max(self._flash_until, t + 0.08)
            if feat.peak > 0.018:
                self._lightning_until = max(self._lightning_until, t + 0.06)

        # NEW: Movement only if auto_move is ON
        if state.auto_move and feat is not None:
            bpm = float(getattr(feat, "bpm", 0.0) or 0.0)
            if bpm > 0.0:
                hz = bpm / 60.0
                self._move_phase_pan += dt * hz * 0.85
                self._move_phase_tilt += dt * hz * 1.00

                bass_boost = float(np.clip(np.log10(1.0 + max(feat.bass, 0.0)) / 4.0, 0.0, 1.0))
                amp_pan = 80.0 + 140.0 * bass_boost
                amp_tilt = 20.0 + 55.0 * bass_boost

                pan_out = int(np.clip(state.pan + np.sin(self._move_phase_pan * 2.0 * np.pi) * amp_pan, 0, 540))
                tilt_out = int(np.clip(state.tilt + np.sin(self._move_phase_tilt * 2.0 * np.pi + 0.7) * amp_tilt, 0, 180))

        fx = (state.fx or "").strip()

        # strobe logic
        strobe_on = True
        if state.strobe_rate > 0:
            self._strobe_phase += dt * float(state.strobe_rate)
            strobe_on = (int(self._strobe_phase) % 2) == 0

        # if dimmer up but rgb black => show white
        if intensity > 0 and base_rgb == (0, 0, 0):
            base_rgb = (255, 255, 255)

        # No FX selected => static if intensity > 0
        if fx == "":
            if intensity <= 0:
                return 0, 0, 0, 0, False, pan_out, tilt_out
            r, g, b = base_rgb
            return intensity, r, g, b, strobe_on, pan_out, tilt_out

        if fx.lower() == "blackout":
            return 0, 0, 0, 0, False, pan_out, tilt_out

        fx_rgb = base_rgb
        fx_int = intensity
        fx_strobe = strobe_on

        if fx.lower() == "flash" or (t < self._flash_until):
            fx_int = 100
            fx_rgb = (255, 255, 255)
            fx_strobe = True

        elif fx.lower() == "lightning" or (t < self._lightning_until):
            flick = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(t * 60.0))
            fx_int = int(100 * flick)
            fx_rgb = (200, 220, 255)
            fx_strobe = True

        elif fx.lower() == "pulse":
            p = 0.5 + 0.5 * np.sin(t * 2.0 * np.pi * 1.2)
            fx_int = int(intensity * p)

        elif fx.lower() == "wave":
            w = 0.7 + 0.3 * np.sin(t * 2.0 * np.pi * 0.25)
            fx_int = int(intensity * w)

        elif fx.lower() == "strobe":
            rate = max(8, int(state.strobe_rate) or 12)
            self._strobe_phase += dt * float(rate)
            fx_strobe = (int(self._strobe_phase) % 2) == 0

        elif fx.lower() == "rainbow":
            hue = (t * 0.15) % 1.0
            fx_rgb = hsv_to_rgb(hue, 1.0, 1.0)

        elif fx.lower() == "chase":
            self._chase_phase += dt * 2.5
            chase = (int(self._chase_phase) % 2) == 0
            fx_int = (intensity if chase else int(intensity * 0.2))

        if fx_int <= 0:
            return 0, 0, 0, 0, False, pan_out, tilt_out

        if state.auto_audio and auto_mix > 0.0:
            out_rgb = self._blend_rgb(base_rgb, fx_rgb, auto_mix)
        else:
            out_rgb = fx_rgb

        return fx_int, out_rgb[0], out_rgb[1], out_rgb[2], fx_strobe, pan_out, tilt_out


# ----------------------------- Simulation Panel -----------------------------
class SimulationWidget(QFrame):
    """
    - Always draw fixture bodies (even if strobe is "off" frame)
    - Only beams/spots blink with strobe
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setMinimumHeight(360)

        self._intensity = 0
        self._rgb = (0, 120, 255)
        self._strobe_on = True
        self._fx_name = "Off"
        self._pan = 270
        self._tilt = 90

    def set_state(self, intensity_0_100: int, r: int, g: int, b: int, strobe_on: bool, fx_name: str, pan: int, tilt: int):
        self._intensity = int(max(0, min(100, intensity_0_100)))
        self._rgb = (int(max(0, min(255, r))), int(max(0, min(255, g))), int(max(0, min(255, b))))
        self._strobe_on = bool(strobe_on)
        self._fx_name = (fx_name or "Off")
        self._pan = int(np.clip(pan, 0, 540))
        self._tilt = int(np.clip(tilt, 0, 180))
        self.update()

    @staticmethod
    def _unit(vx: float, vy: float) -> Tuple[float, float]:
        n = float(np.hypot(vx, vy))
        if n <= 1e-6:
            return 0.0, 0.0
        return vx / n, vy / n

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing)

        R = self.rect().adjusted(14, 14, -14, -14)

        p.fillRect(R, QColor(10, 12, 16, 230))
        vign = QRadialGradient(QPointF(R.center().x(), R.center().y()), float(max(R.width(), R.height())) * 0.7)
        vign.setColorAt(0.0, QColor(25, 28, 33, 120))
        vign.setColorAt(1.0, QColor(0, 0, 0, 220))
        p.fillRect(R, QBrush(vign))

        p.setPen(QPen(QColor(214, 217, 222, 180), 1))
        p.drawText(R.left() + 10, R.top() + 22, f"Simulation — FX: {self._fx_name}   |   Pan: {self._pan}°   Tilt: {self._tilt}°")

        intensity = self._intensity / 100.0
        r, g, b = self._rgb
        has_value = (self._intensity > 0) and not (r == 0 and g == 0 and b == 0)

        # scene geometry
        truss_y = int(R.top() + R.height() * 0.18)
        floor_y = int(R.top() + R.height() * 0.86)

        p.setPen(QPen(QColor(60, 70, 85, 90), 2))
        p.drawLine(R.left() + 30, truss_y, R.right() - 30, truss_y)
        p.setPen(QPen(QColor(60, 70, 85, 60), 1))
        p.drawLine(R.left() + 30, floor_y, R.right() - 30, floor_y)

        fx1_x = int(R.left() + R.width() * 0.38)
        fx2_x = int(R.left() + R.width() * 0.62)

        pan_norm = (self._pan - 270.0) / 270.0
        tilt_norm = self._tilt / 180.0

        x_range = float(R.width() * 0.28)
        y_range = float(R.height() * 0.50)

        spot_y = float(floor_y - (1.0 - tilt_norm) * y_range)
        spot_y = float(np.clip(spot_y, R.top() + R.height() * 0.35, floor_y))

        spot1_x = float(np.clip(fx1_x + pan_norm * x_range, R.left() + 60, R.right() - 60))
        spot2_x = float(np.clip(fx2_x + pan_norm * x_range, R.left() + 60, R.right() - 60))

        def draw_head(fx_x: int, target_x: float, target_y: float, draw_beam: bool):
            base_w = int(R.width() * 0.08)
            base_h = int(R.height() * 0.05)
            yoke_h = int(R.height() * 0.07)
            head_r = int(R.width() * 0.028)

            base = QRect(fx_x - base_w // 2, truss_y - base_h // 2, base_w, base_h)
            p.setPen(Qt.PenStyle.NoPen if QT6 else Qt.NoPen)
            p.setBrush(QColor(28, 32, 38, 220))
            p.drawRoundedRect(base, 10, 10)

            p.setPen(QPen(QColor(90, 100, 115, 140), 5, Qt.PenStyle.SolidLine if QT6 else Qt.SolidLine,
                          Qt.PenCapStyle.RoundCap if QT6 else Qt.RoundCap))
            p.drawLine(fx_x - base_w // 3, truss_y + base_h // 2, fx_x - base_w // 3, truss_y + base_h // 2 + yoke_h)
            p.drawLine(fx_x + base_w // 3, truss_y + base_h // 2, fx_x + base_w // 3, truss_y + base_h // 2 + yoke_h)

            lens_x = float(fx_x)
            lens_y = float(truss_y + base_h // 2 + yoke_h)

            dx = float(target_x - lens_x)
            dy = float(target_y - lens_y)
            ux, uy = self._unit(dx, dy)

            head_cx = lens_x + ux * head_r * 0.6
            head_cy = lens_y + uy * head_r * 0.6

            p.setPen(Qt.PenStyle.NoPen if QT6 else Qt.NoPen)
            p.setBrush(QColor(35, 40, 48, 230))
            p.drawEllipse(QPointF(head_cx, head_cy), head_r, head_r)

            # Lens glow always visible (but smaller when "beam off")
            glow_int = (0.35 + 0.65 * intensity) if has_value else 0.15
            if not draw_beam:
                glow_int *= 0.35

            lens_a = int(70 + 160 * glow_int)
            p.setBrush(QColor(r, g, b, lens_a))
            p.drawEllipse(QPointF(head_cx, head_cy), head_r * 0.55, head_r * 0.55)

            p.setPen(QPen(QColor(210, 220, 235, 90), 3, Qt.PenStyle.SolidLine if QT6 else Qt.SolidLine,
                          Qt.PenCapStyle.RoundCap if QT6 else Qt.RoundCap))
            p.drawLine(QPointF(head_cx, head_cy), QPointF(head_cx + ux * head_r * 1.2, head_cy + uy * head_r * 1.2))

            if not draw_beam:
                return

            dist = float(np.hypot(dx, dy))
            beam_w0 = float(head_r * 0.45)
            beam_w1 = float(np.clip(dist * 0.09, 20.0, 120.0))

            px, py = -uy, ux

            p0 = QPointF(lens_x + px * beam_w0, lens_y + py * beam_w0)
            p1 = QPointF(lens_x - px * beam_w0, lens_y - py * beam_w0)
            p2 = QPointF(target_x - px * beam_w1, target_y - py * beam_w1)
            p3 = QPointF(target_x + px * beam_w1, target_y + py * beam_w1)
            poly = QPolygonF([p0, p1, p2, p3])

            grad = QLinearGradient(QPointF(lens_x, lens_y), QPointF(target_x, target_y))
            base_a = int(22 + 85 * intensity)
            mid_a = int(14 + 55 * intensity)
            end_a = int(6 + 30 * intensity)
            grad.setColorAt(0.0, QColor(r, g, b, base_a))
            grad.setColorAt(0.6, QColor(r, g, b, mid_a))
            grad.setColorAt(1.0, QColor(r, g, b, end_a))
            p.setPen(Qt.PenStyle.NoPen if QT6 else Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawPolygon(poly)

            spot_rad = float(np.clip(45.0 + dist * 0.05, 50.0, 140.0))
            spot = QRadialGradient(QPointF(target_x, target_y), spot_rad)
            spot.setColorAt(0.0, QColor(255, 255, 255, int(90 + 120 * intensity)))
            spot.setColorAt(0.25, QColor(r, g, b, int(60 + 120 * intensity)))
            spot.setColorAt(1.0, QColor(r, g, b, 0))
            p.setBrush(QBrush(spot))
            p.drawEllipse(QPointF(target_x, target_y), spot_rad, spot_rad)

        if not has_value:
            p.setPen(QPen(QColor(120, 130, 145, 140), 1))
            p.drawText(R.left() + 10, R.top() + 44, "OFF (raise DIMMER or choose a color)")
            # still draw bodies (no beam)
            draw_head(fx1_x, spot1_x, spot_y, draw_beam=False)
            draw_head(fx2_x, spot2_x, spot_y, draw_beam=False)
            p.end()
            return

        # draw_beam is what strobes — not the head body
        draw_beam = bool(self._strobe_on)

        draw_head(fx1_x, spot1_x, spot_y, draw_beam=draw_beam)
        draw_head(fx2_x, spot2_x, spot_y, draw_beam=draw_beam)

        p.end()


# ----------------------------- Presets -----------------------------
@dataclass
class SnapshotPreset:
    name: str
    state: ControlState
    duration_sec: float = 0.0


# ----------------------------- Single Panel UI -----------------------------
class SinglePanel(QWidget):
    import_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    replay_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state = ControlState()

        self.is_recording = False
        self.can_save = False
        self._rec_start_monotonic: Optional[float] = None
        self._rec_elapsed_sec: float = 0.0

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        split = QSplitter(Qt.Orientation.Horizontal if QT6 else Qt.Horizontal)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # -------- Left panel --------
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)

        top_panel = Panel("Show Controls")
        left_lay.addWidget(top_panel, 0)

        status_row = QHBoxLayout()
        self.time_lbl = QLabel("00:00")
        self.time_lbl.setFont(QFont("Arial", 12, 700))
        self.time_lbl.setObjectName("PanelTitle")
        self.status_lbl = QLabel("FX: Off")
        self.status_lbl.setObjectName("Subtle")
        status_row.addWidget(self.time_lbl)
        status_row.addSpacing(10)
        status_row.addWidget(self.status_lbl)
        status_row.addStretch(1)
        top_panel.body.addLayout(status_row)

        rec_row = QHBoxLayout()
        rec_row.setSpacing(10)

        self.btn_record = QPushButton("RECORD")
        self.btn_stoprec = QPushButton("STOP")
        self.btn_stoprec.setObjectName("Danger")
        self.btn_save = QPushButton("SAVE")
        self.btn_load = QPushButton("LOAD")

        for b in (self.btn_record, self.btn_stoprec, self.btn_save, self.btn_load):
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        rec_row.addWidget(self.btn_record)
        rec_row.addWidget(self.btn_stoprec)
        rec_row.addWidget(self.btn_save)
        rec_row.addWidget(self.btn_load)
        top_panel.body.addLayout(rec_row)

        fx_panel = Panel("Built-in FX (click again to turn off)")
        left_lay.addWidget(fx_panel, 0)

        self.fx_group = QButtonGroup(self)
        self.fx_group.setExclusive(False)
        self._active_fx_btn: Optional[QToolButton] = None

        fx_grid = QGridLayout()
        fx_grid.setSpacing(10)

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
            fx_grid.addWidget(btn, i // 4, i % 4)

        fx_panel.body.addLayout(fx_grid)

        for btn in self.fx_buttons.values():
            btn.clicked.connect(lambda checked, b=btn: self._on_fx_clicked(b, checked))

        slider_panel = Panel("Sliders")
        left_lay.addWidget(slider_panel, 1)

        # ---- CENTERED + EVEN slider bank ----
        slider_box = QWidget()
        slider_box_lay = QHBoxLayout(slider_box)
        slider_box_lay.setContentsMargins(0, 0, 0, 0)
        slider_box_lay.setSpacing(12)

        def make_vslider(title: str, rng: Tuple[int, int], val: int, tick_interval: int = 0) -> Tuple[QWidget, QSlider, QLabel]:
            well = SliderWell()
            well.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            well.setFixedSize(92, 340)  # <--- EVEN size (w,h)

            v = QVBoxLayout(well)
            v.setContentsMargins(12, 12, 12, 12)
            v.setSpacing(8)

            lab = QLabel(title)
            lab.setObjectName("Subtle")
            lab.setAlignment(Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter)

            s = QSlider(Qt.Orientation.Vertical if QT6 else Qt.Vertical)
            s.setRange(rng[0], rng[1])
            s.setValue(val)
            s.setMinimumHeight(250)
            s.setTickPosition(QSlider.TickPosition.NoTicks if QT6 else QSlider.NoTicks)
            if tick_interval > 0:
                s.setTickInterval(tick_interval)

            ro = QLabel(str(val))
            ro.setObjectName("Subtle")
            ro.setAlignment(Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter)

            v.addWidget(lab)
            v.addWidget(s, 1)
            v.addWidget(ro)
            return well, s, ro

        w_dim, self.s_dim, self.ro_dim = make_vslider("DIMMER", (0, 100), self.state.intensity)
        w_r, self.s_r, self.ro_r = make_vslider("RED", (0, 255), self.state.r)
        w_g, self.s_g, self.ro_g = make_vslider("GREEN", (0, 255), self.state.g)
        w_b, self.s_b, self.ro_b = make_vslider("BLUE", (0, 255), self.state.b)
        w_st, self.s_strobe, self.ro_st = make_vslider("STROBE", (0, 20), self.state.strobe_rate)
        w_pan, self.s_pan, self.ro_pan = make_vslider("PAN", (0, 540), self.state.pan)
        w_tilt, self.s_tilt, self.ro_tilt = make_vslider("TILT", (0, 180), self.state.tilt)

        for w in (w_dim, w_r, w_g, w_b, w_st, w_pan, w_tilt):
            slider_box_lay.addWidget(w)

        slider_outer = QHBoxLayout()
        slider_outer.setContentsMargins(0, 0, 0, 0)
        slider_outer.addWidget(
            slider_box,
            0,
            Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter
        )
        slider_panel.body.addLayout(slider_outer)

        # Toggles
        opts_row = QHBoxLayout()
        opts_row.setSpacing(18)

        self.auto_toggle = ToggleSwitch("AUTO (Audio-driven)")
        self.auto_toggle.setChecked(False)

        self.auto_move_toggle = ToggleSwitch("AUTO PAN/TILT (BPM)")   # NEW
        self.auto_move_toggle.setChecked(False)

        self.dmx_enable = ToggleSwitch("DMX Output")
        self.dmx_enable.setChecked(False)

        opts_row.addWidget(self.auto_toggle)
        opts_row.addWidget(self.auto_move_toggle)
        opts_row.addWidget(self.dmx_enable)
        opts_row.addStretch(1)
        slider_panel.body.addLayout(opts_row)

        pname_row = QHBoxLayout()
        pname_row.setSpacing(10)
        self.preset_name = QLineEdit(self.state.preset_name)
        self.preset_name.setPlaceholderText("Preset name (for SAVE/LOAD)")
        pname_row.addWidget(QLabel("Preset:"))
        pname_row.addWidget(self.preset_name, 1)
        slider_panel.body.addLayout(pname_row)

        left_lay.addStretch(1)

        # -------- Right panel --------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(10)

        sim_panel = Panel("Light Simulation")
        right_lay.addWidget(sim_panel, 1)

        self.sim = SimulationWidget()
        sim_panel.body.addWidget(self.sim, 1)

        audio_panel = Panel("Music / Audio")
        right_lay.addWidget(audio_panel, 0)

        bpm_row = QHBoxLayout()
        bpm_row.setSpacing(10)
        self.bpm_lbl = QLabel("BPM: --")
        self.bpm_lbl.setObjectName("Subtle")
        bpm_row.addWidget(self.bpm_lbl)
        bpm_row.addStretch(1)
        audio_panel.body.addLayout(bpm_row)

        self.visualizer = SimpleVisualizer(nbars=64)
        audio_panel.body.addWidget(self.visualizer)

        audio_row = QHBoxLayout()
        audio_row.setSpacing(10)
        self.btn_play = QPushButton("PLAY")
        self.btn_stop = QPushButton("STOP")
        self.btn_replay = QPushButton("REPLAY")
        self.btn_import = QPushButton("IMPORT")

        self.btn_play.setObjectName("Primary")
        self.btn_stop.setObjectName("Danger")

        audio_row.addWidget(self.btn_play)
        audio_row.addWidget(self.btn_stop)
        audio_row.addWidget(self.btn_replay)
        audio_row.addStretch(1)
        audio_row.addWidget(self.btn_import)
        audio_panel.body.addLayout(audio_row)

        self.wav_lbl = QLabel("WAV: (none)")
        self.wav_lbl.setObjectName("Subtle")
        self.wav_lbl.setWordWrap(True)
        audio_panel.body.addWidget(self.wav_lbl)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)

        # ---- Wiring ----
        for s in (self.s_dim, self.s_r, self.s_g, self.s_b, self.s_strobe, self.s_pan, self.s_tilt):
            s.valueChanged.connect(self._on_controls_changed)

        self.auto_toggle.toggled.connect(lambda _: self._on_controls_changed())
        self.auto_move_toggle.toggled.connect(lambda _: self._on_controls_changed())
        self.dmx_enable.toggled.connect(lambda _: self._on_controls_changed())
        self.preset_name.textChanged.connect(self._on_controls_changed)

        self.btn_import.clicked.connect(self.import_clicked.emit)
        self.btn_play.clicked.connect(self.play_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_replay.clicked.connect(self.replay_clicked.emit)

        self.btn_record.clicked.connect(self._record_start)
        self.btn_stoprec.clicked.connect(self._record_stop)
        self.btn_save.clicked.connect(self._save_preset_dialog)
        self.btn_load.clicked.connect(self._load_preset_dialog)

        self._sync_record_ui()
        self._on_controls_changed()

    # ---------- FX toggle-off ----------
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

    # ---------- Controls ----------
    def _on_controls_changed(self, *_):
        self.state.intensity = int(self.s_dim.value())
        self.state.r = int(self.s_r.value())
        self.state.g = int(self.s_g.value())
        self.state.b = int(self.s_b.value())
        self.state.strobe_rate = int(self.s_strobe.value())
        self.state.pan = int(self.s_pan.value())
        self.state.tilt = int(self.s_tilt.value())

        self.state.auto_audio = bool(self.auto_toggle.isChecked())
        self.state.auto_move = bool(self.auto_move_toggle.isChecked())
        self.state.preset_name = self.preset_name.text().strip() or "Untitled"

        # lock pan/tilt sliders if auto_move is on
        self.s_pan.setEnabled(not self.state.auto_move)
        self.s_tilt.setEnabled(not self.state.auto_move)

        self.ro_dim.setText(str(self.state.intensity))
        self.ro_r.setText(str(self.state.r))
        self.ro_g.setText(str(self.state.g))
        self.ro_b.setText(str(self.state.b))
        self.ro_st.setText(str(self.state.strobe_rate))
        self.ro_pan.setText(str(self.state.pan))
        self.ro_tilt.setText(str(self.state.tilt))

    # ---------- Recording / Timecode ----------
    def _sync_record_ui(self):
        self.btn_save.setEnabled(self.can_save)
        self.btn_record.setEnabled(not self.is_recording)
        self.btn_stoprec.setEnabled(self.is_recording)

        if self.is_recording:
            self.btn_record.setObjectName("Primary")
        else:
            self.btn_record.setObjectName("")
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)

    def _record_start(self):
        self.is_recording = True
        self.can_save = False
        self._rec_start_monotonic = time.monotonic()
        self._rec_elapsed_sec = 0.0
        self.status_lbl.setText("Recording… (press STOP to finish)")
        self._sync_record_ui()

    def _record_stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.can_save = True
        if self._rec_start_monotonic is not None:
            self._rec_elapsed_sec = max(0.0, time.monotonic() - self._rec_start_monotonic)
        self._rec_start_monotonic = None

        if self.state.fx:
            self.status_lbl.setText(f"Stopped (ready to SAVE) — FX: {self.state.fx}")
        else:
            self.status_lbl.setText("Stopped (ready to SAVE) — FX: Off")
        self._sync_record_ui()

    def update_timecode_label(self):
        if self.is_recording and self._rec_start_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - self._rec_start_monotonic)
        else:
            elapsed = self._rec_elapsed_sec

        mm = int(elapsed // 60)
        ss = int(elapsed % 60)
        self.time_lbl.setText(f"{mm:02d}:{ss:02d}")

    def recorded_duration_sec(self) -> float:
        if self.is_recording and self._rec_start_monotonic is not None:
            return max(0.0, time.monotonic() - self._rec_start_monotonic)
        return float(self._rec_elapsed_sec)

    # ---------- Presets ----------
    def _save_preset_dialog(self):
        if not self.can_save:
            QMessageBox.information(self, "Not ready", "Click RECORD, then STOP, then you can SAVE.")
            return

        preset = SnapshotPreset(
            name=self.state.preset_name,
            state=self.state,
            duration_sec=self.recorded_duration_sec(),
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", os.path.join(os.getcwd(), f"{preset.name}.json"), "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            data = asdict(preset)
            data["state"] = asdict(preset.state)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.status_lbl.setText(f"Saved preset: {os.path.basename(path)}")
            self.can_save = False
            self._sync_record_ui()
        except Exception as e:
            QMessageBox.critical(self, "Preset Save Error", str(e))

    def _load_preset_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", os.getcwd(), "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = data.get("state", data)

            fx = str(st.get("fx", "")).title()
            if fx not in FX_LIST:
                fx = ""

            for b in self.fx_buttons.values():
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)

            if fx and fx in self.fx_buttons:
                self.fx_buttons[fx].blockSignals(True)
                self.fx_buttons[fx].setChecked(True)
                self.fx_buttons[fx].blockSignals(False)
                self._active_fx_btn = self.fx_buttons[fx]
                self.state.fx = fx
                self.status_lbl.setText(f"Loaded preset — FX: {fx}")
            else:
                self._active_fx_btn = None
                self.state.fx = ""
                self.status_lbl.setText("Loaded preset — FX: Off")

            self.s_dim.setValue(int(st.get("intensity", 80)))
            self.s_r.setValue(int(st.get("r", 0)))
            self.s_g.setValue(int(st.get("g", 120)))
            self.s_b.setValue(int(st.get("b", 255)))
            self.s_strobe.setValue(int(st.get("strobe_rate", 0)))
            self.s_pan.setValue(int(st.get("pan", 270)))
            self.s_tilt.setValue(int(st.get("tilt", 90)))

            self.auto_toggle.setChecked(bool(st.get("auto_audio", False)))
            self.auto_move_toggle.setChecked(bool(st.get("auto_move", False)))

            name = os.path.splitext(os.path.basename(path))[0]
            self.preset_name.setText(str(st.get("preset_name", name)) or name)

            self.is_recording = False
            self.can_save = False
            self._rec_start_monotonic = None
            self._rec_elapsed_sec = float(data.get("duration_sec", 0.0) or 0.0)
            self._sync_record_ui()

        except Exception as e:
            QMessageBox.critical(self, "Preset Load Error", str(e))

    # ---------- Audio UI hooks ----------
    def set_loaded_file(self, path: str):
        self.wav_lbl.setText(f"WAV: {path}")

    def update_visualizer(self, feat: AudioFeatures):
        self.visualizer.update_from_spectrum(feat.spectrum)
        if feat.bpm and feat.bpm > 0.0:
            self.bpm_lbl.setText(f"BPM: {feat.bpm:0.0f}")
        else:
            self.bpm_lbl.setText("BPM: --")


# ----------------------------- Main Window -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lighting FX + Audio Visualizer + Simulation (Single Panel)")
        self.resize(1400, 820)

        self.ui = SinglePanel()
        self.setCentralWidget(self.ui)

        self.output = OutputEngine()
        self.output_cfg = OutputConfig(enabled=False)
        self.output.apply_config(self.output_cfg)

        self.output_timer = QTimer(self)
        self.output_timer.setInterval(10)
        self.output_timer.timeout.connect(self.output.tick)
        self.output_timer.start()

        self.file_thread = QThread(self)
        self.file_worker = AudioFileWorker()
        self.file_worker.moveToThread(self.file_thread)
        self.file_thread.start()

        self.file_worker.features.connect(self._on_audio_features)
        self.file_worker.error.connect(self._on_audio_error)

        self.ui.import_clicked.connect(self._import_wav)
        self.ui.play_clicked.connect(self._play)
        self.ui.stop_clicked.connect(self._stop)
        self.ui.replay_clicked.connect(self._replay)

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
        self.ui.status_lbl.setText("Playing audio")
        QTimer.singleShot(0, self.file_worker.play)

    def _stop(self):
        self.ui.status_lbl.setText("Stopped")
        QTimer.singleShot(0, self.file_worker.stop)

    def _replay(self):
        if not self._wav_path:
            QMessageBox.information(self, "No WAV", "Click IMPORT first.")
            return
        self.ui.status_lbl.setText("Replay audio")
        QTimer.singleShot(0, self.file_worker.replay)

    def _on_audio_features(self, feat_obj):
        feat: AudioFeatures = feat_obj
        self._latest_feat = feat
        self.ui.update_visualizer(feat)

    def _on_audio_error(self, msg: str):
        QMessageBox.critical(self, "Audio Error", msg)

    def _frame_tick(self):
        now = time.monotonic()
        dt = max(0.001, now - self._last_t)
        self._last_t = now

        self.ui.update_timecode_label()

        intensity, r, g, b, st_on, pan_out, tilt_out = self.fx.tick(self.ui.state, self._latest_feat, dt, now)
        fx_name = self.ui.state.fx if self.ui.state.fx else "Off"
        self.ui.sim.set_state(intensity, r, g, b, st_on, fx_name, pan_out, tilt_out)

        # DMX mapping: [Dimmer, R, G, B, Strobe, Pan, Tilt] (0..100)
        if self.ui.dmx_enable.isChecked():
            dim = intensity
            fr = int((r / 255.0) * 100)
            fg = int((g / 255.0) * 100)
            fb = int((b / 255.0) * 100)
            fst = int(min(100, (self.ui.state.strobe_rate / 20.0) * 100))
            fpan = int(min(100, (pan_out / 540.0) * 100))
            ftilt = int(min(100, (tilt_out / 180.0) * 100))

            if not st_on and self.ui.state.strobe_rate > 0:
                dim = 0

            self.output.apply_config(OutputConfig(enabled=True))
            self.output.set_channels_from_faders([dim, fr, fg, fb, fst, fpan, ftilt])
        else:
            self.output.apply_config(OutputConfig(enabled=False))


# ----------------------------- Entrypoint -----------------------------
def main():
    app = QApplication([])
    app.setStyleSheet(DARK_QSS)
    w = MainWindow()
    w.show()
    if QT6:
        app.exec()
    else:
        app.exec_()


if __name__ == "__main__":
    main()
