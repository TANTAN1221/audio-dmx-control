from __future__ import annotations

import os
import time
from typing import Optional

from .qt_compat import (
    QApplication, QMainWindow, QThread, QTimer, QMessageBox, QFileDialog, QEvent, Qt
)
from .audio.worker import AudioFileWorker
from .audio.features import AudioFeatures
from .lighting.fx import FxEngine
from .output import OutputEngine, OutputConfig
from .ui.single_panel import SinglePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lighting FX + Audio Visualizer + Simulation (Single Panel)")
        self.setMinimumSize(854, 480)

        # Remove maximize button (optional) - since we enforce maximized anyway.
        try:
            self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)
        except Exception:
            try:
                self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
            except Exception:
                pass

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
        self.file_worker.position.connect(self._on_audio_position)
        self.file_worker.beatsReady.connect(self._on_beats_ready)
        self.file_worker.error.connect(self._on_audio_error)

        self.ui.import_clicked.connect(self._import_wav)
        self.ui.save_clicked.connect(self._save_preset)
        self.ui.load_clicked.connect(self._load_preset)
        self.ui.play_clicked.connect(self._play)
        self.ui.stop_clicked.connect(self._stop)
        self.ui.replay_clicked.connect(self._replay)
        self.ui.seek_ratio.connect(self._seek_ratio)

        self.fx = FxEngine()
        self._prev_audio_sec: float = 0.0
        self._scheduled_fx_end_sec: float = 0.0
        self._latest_feat = AudioFeatures()
        self._last_t = time.monotonic()

        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(33)
        self.frame_timer.timeout.connect(self._frame_tick)
        self.frame_timer.start()

        self._wav_path: Optional[str] = None

        # Lock to maximized after first show (and keep it that way)
        QTimer.singleShot(0, self._lock_maximized)

    def _lock_maximized(self):
        """
        Force the window to stay maximized and non-resizable.
        """
        self.showMaximized()

        # Make the window fixed-size to the available screen area.
        try:
            screen = self.screen()
            if screen is None:
                screen = QApplication.primaryScreen()
            geo = screen.availableGeometry()
            self.setGeometry(geo)
            self.setFixedSize(geo.size())
        except Exception:
            # If something goes wrong, at least keep it maximized.
            pass

    def changeEvent(self, event):
        # If user tries to restore/unmaximize, re-lock.
        try:
            if event.type() == QEvent.Type.WindowStateChange and not self.isMaximized():
                QTimer.singleShot(0, self._lock_maximized)
        except Exception:
            pass
        super().changeEvent(event)

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

    
    def _on_beats_ready(self, bpm: float, beats):
        try:
            self.ui.set_beats(float(bpm or 0.0), list(beats or []))
        except Exception:
            pass

    def _on_audio_position(self, cur_sec: float, total_sec: float):
        # Update UI
        self.ui.set_player_position(cur_sec, total_sec)

        # Timeline cues -> scheduled FX
        # If user seeks backwards, reset cue scanning.
        if cur_sec + 0.001 < self._prev_audio_sec:
            self._prev_audio_sec = cur_sec
            self._scheduled_fx_end_sec = 0.0
            self.ui.state.scheduled_fx = ""
            return

        # Trigger any cues that fall in (prev, cur]
        hits = self.ui.timeline.take_triggers(self._prev_audio_sec, cur_sec)
        if hits:
            cue = hits[-1]  # most recent wins
            self.ui.state.scheduled_fx = cue.fx
            self._scheduled_fx_end_sec = float(cur_sec) + float(cue.dur)

        # Expire scheduled FX
        if self.ui.state.scheduled_fx and cur_sec >= self._scheduled_fx_end_sec:
            self.ui.state.scheduled_fx = ""

        self._prev_audio_sec = cur_sec


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


    def _save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "", "Preset JSON (*.json)")
        if not path:
            return
        try:
            data = self.ui.export_preset()
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save preset", f"Failed to save: {e}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset JSON (*.json)")
        if not path:
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ui.import_preset(data)
        except Exception as e:
            QMessageBox.critical(self, "Load preset", f"Failed to load: {e}")
