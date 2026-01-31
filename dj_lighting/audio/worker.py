from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from ..qt_compat import QObject, pyqtSignal
from .features import AudioFeatures, compute_features_from_mono, estimate_bpm_and_beats


class AudioFileWorker(QObject):
    """
    Plays a WAV and emits AudioFeatures + current position at ~30 Hz.
    Runs in its own QThread.
    """
    features = pyqtSignal(object)
    position = pyqtSignal(float, float)  # current_sec, total_sec
    beatsReady = pyqtSignal(float, object)  # bpm, beat_times(list[float])
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

            # Offline beat grid estimation (for timeline markers)
            try:
                mono = self._audio.mean(axis=1)
                bpm0, beats = estimate_bpm_and_beats(mono, self._sr)
                self.beatsReady.emit(float(bpm0), beats)
            except Exception:
                pass

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
