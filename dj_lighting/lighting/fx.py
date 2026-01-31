from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..audio.features import AudioFeatures

FX_LIST = ["Wave", "Pulse", "Flash", "Strobe", "Lightning", "Rainbow", "Chase", "Blackout"]


@dataclass
class ControlState:
    fx: str = ""  # manual FX toggle
    scheduled_fx: str = ""  # FX triggered from timeline cues (temporary)
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
        # ---- base controls ----
        intensity = int(np.clip(state.intensity, 0, 100))
        r, g, b = int(state.r), int(state.g), int(state.b)

        pan_out = int(np.clip(state.pan, 0, 540))
        tilt_out = int(np.clip(state.tilt, 0, 180))

        # ---- audio-driven base (AUTO) ----
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

        # ---- auto move (BPM) ----
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

        # ---- FX ----
        fx = ((getattr(state, "scheduled_fx", "") or getattr(state, "fx", "")) or "").strip().lower()
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

        # manual strobe slider also works outside STROBE FX
        if fx != "strobe" and state.strobe_rate > 0:
            self._strobe_phase += dt * float(state.strobe_rate)
            strobe_on = (int(self._strobe_phase) % 2) == 0

        if intensity > 0 and (r, g, b) == (0, 0, 0):
            r, g, b = 255, 255, 255

        if intensity <= 0:
            return 0, 0, 0, 0, False, pan_out, tilt_out

        return intensity, r, g, b, bool(strobe_on), pan_out, tilt_out
