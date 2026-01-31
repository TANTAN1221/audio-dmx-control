from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
    """
    Compute simple audio features (RMS, peak delta, band means, 24-band spectrum).
    Returns (features, updated_prev_rms).
    """
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


def estimate_bpm_and_beats(mono: np.ndarray, sr: int) -> Tuple[float, List[float]]:
    """Estimate BPM and beat times (seconds) from mono audio.

    Lightweight, dependency-free approach:
    - RMS/onset envelope over short frames
    - Autocorrelation to find dominant periodicity (40-220 BPM)
    - Peak-pick beats from the envelope with a minimum distance constraint

    Returns (bpm, beat_times_seconds).
    """
    if mono is None or len(mono) < sr // 4 or sr <= 0:
        return 0.0, []

    mono = mono.astype(np.float32, copy=False)
    mono = mono - float(np.mean(mono))

    frame = 1024
    hop = 512
    n = len(mono)
    n_frames = 1 + max(0, (n - frame) // hop)
    if n_frames < 16:
        return 0.0, []

    # RMS envelope
    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = i * hop
        b = a + frame
        x = mono[a:b]
        env[i] = float(np.sqrt(np.mean(x * x) + 1e-12))

    # Emphasize onsets (simple high-pass via moving average subtraction)
    k = 16
    ma = np.convolve(env, np.ones(k, dtype=np.float32) / k, mode="same")
    onset = np.maximum(0.0, env - ma)
    onset = onset / (float(np.max(onset)) + 1e-9)

    # Autocorrelation in tempo band
    min_bpm, max_bpm = 40.0, 220.0
    min_lag = int((60.0 * sr) / (max_bpm * hop))
    max_lag = int((60.0 * sr) / (min_bpm * hop))
    max_lag = min(max_lag, n_frames - 2)
    if max_lag <= min_lag + 2:
        return 0.0, []

    x = onset - float(np.mean(onset))
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    ac[:min_lag] = 0.0
    ac[max_lag+1:] = 0.0

    lag = int(np.argmax(ac))
    if lag <= 0 or not np.isfinite(ac[lag]) or float(ac[lag]) <= 1e-6:
        return 0.0, []

    bpm = float(np.clip(60.0 * sr / (lag * hop), min_bpm, max_bpm))

    # Peak-pick beats from onset envelope
    # Minimum distance ~ 70% of the estimated period
    min_dist = max(1, int(0.7 * lag))
    thr = float(np.mean(onset) + 0.7 * np.std(onset))
    peaks: List[int] = []
    last = -10**9
    for i in range(1, n_frames - 1):
        if onset[i] >= thr and onset[i] >= onset[i-1] and onset[i] >= onset[i+1]:
            if i - last >= min_dist:
                peaks.append(i)
                last = i

    # If peak picking failed (quiet track), generate a grid from the tempo.
    if len(peaks) < 4:
        period_sec = (60.0 / bpm) if bpm > 0 else 0.0
        if period_sec <= 0:
            return 0.0, []
        total_sec = float(n) / float(sr)
        t = 0.0
        beats = []
        while t <= total_sec + 1e-6:
            beats.append(t)
            t += period_sec
        return bpm, beats

    beat_times = [float(i * hop) / float(sr) for i in peaks]
    return bpm, beat_times
