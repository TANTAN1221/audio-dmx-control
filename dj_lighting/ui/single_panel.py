from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..audio.features import AudioFeatures
from ..lighting.fx import ControlState
from ..qt_compat import (
    Qt,
    pyqtSignal,
    QWidget,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QSizePolicy,
    QButtonGroup,
    QSlider,
    QLineEdit,
    QMessageBox,
    QColorDialog,
    QBoxLayout,
    QTimer,
    QFont,
)
from ..qt_compat import hbox_dir_left_to_right, hbox_dir_top_to_bottom, orientation_horizontal
from .panels import Panel
from .controls import ToggleSwitch, SliderCard, FxDragButton
from .visualizer import SimpleVisualizer
from .simulation import SimulationWidget
from .timeline import FxTimeline
from .utils import fmt_time


class SinglePanel(QWidget):
    import_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    load_clicked = pyqtSignal()
    play_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    replay_clicked = pyqtSignal()
    seek_ratio = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state = ControlState()

        self._recording = False
        self._cur_sec = 0.0
        self._last_scale = -1.0
        self._user_scrubbing = False
        self._syncing_color = False

        # Responsive baseline
        self._base_w = 1366
        self._base_h = 768

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ============================================================
        # Top row: Timeline (with beat grid) + Transport beside it
        # ============================================================
        self.top_row = QWidget()
        self.top_row_lay = QHBoxLayout(self.top_row)
        self.top_row_lay.setContentsMargins(0, 0, 0, 0)
        self.top_row_lay.setSpacing(10)

        # Timeline panel
        self.timeline_panel = Panel("Audio Timeline")
        self.timeline_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.timeline = FxTimeline()
        self.timeline_panel.body.addWidget(self.timeline)

        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(10)
        self.time_cur = QLabel("00:00")
        self.time_cur.setObjectName("Subtle")
        self.time_tot = QLabel("00:00")
        self.time_tot.setObjectName("Subtle")

        self.scrub = QSlider(orientation_horizontal())
        self.scrub.setRange(0, 1000)
        self.scrub.setValue(0)
        self.scrub.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        scrub_row.addWidget(self.time_cur)
        scrub_row.addWidget(self.scrub, 1)
        scrub_row.addWidget(self.time_tot)
        self.timeline_panel.body.addLayout(scrub_row)

        # Transport panel (beside timeline)
        self.transport_panel = Panel("Transport")
        self.transport_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        tp = self.transport_panel.body

        # Status + time
        top_status = QVBoxLayout()
        top_status.setSpacing(6)

        self.time_lbl = QLabel("00:00")
        self.time_lbl.setFont(QFont("Segoe UI", 14, 800))
        self.time_lbl.setObjectName("PanelTitle")

        self.status_lbl = QLabel("Stopped")
        self.status_lbl.setObjectName("Subtle")

        top_status.addWidget(self.time_lbl)
        top_status.addWidget(self.status_lbl)
        tp.addLayout(top_status)

        # Row 1: playback/import
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        self.btn_play = QPushButton("PLAY")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("Danger")
        self.btn_replay = QPushButton("REPLAY")
        self.btn_import = QPushButton("IMPORT")
        for b in [self.btn_play, self.btn_stop, self.btn_replay, self.btn_import]:
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row1.addWidget(self.btn_play)
        btn_row1.addWidget(self.btn_stop)
        btn_row1.addWidget(self.btn_replay)
        btn_row1.addWidget(self.btn_import)
        tp.addLayout(btn_row1)

        # Row 2: record
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(8)
        self.btn_record = QPushButton("RECORD")
        self.btn_stoprec = QPushButton("STOP REC")
        self.btn_stoprec.setObjectName("Danger")
        for b in [self.btn_record, self.btn_stoprec]:
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_row2.addWidget(self.btn_record)
        btn_row2.addWidget(self.btn_stoprec)
        tp.addLayout(btn_row2)

        self.wav_lbl = QLabel("WAV: (none)")
        self.wav_lbl.setObjectName("Subtle")
        self.wav_lbl.setWordWrap(True)
        tp.addWidget(self.wav_lbl)

        self.top_row_lay.addWidget(self.timeline_panel, 1)
        self.top_row_lay.addWidget(self.transport_panel, 0)
        root.addWidget(self.top_row, 0)

        # Scrub signals
        self.scrub.sliderPressed.connect(self._scrub_start)
        self.scrub.sliderReleased.connect(self._scrub_commit)

        # ============================================================
        # Main content splitter (Left: FX+Sliders | Right: Sim+Preset+Audio)
        # ============================================================
        self.split = QSplitter(Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        root.addWidget(self.split, 1)

        # ---------------- Left side ----------------
        self.left = QWidget()
        self.left_lay = QVBoxLayout(self.left)
        self.left_lay.setContentsMargins(0, 0, 0, 0)
        self.left_lay.setSpacing(10)

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
            btn = FxDragButton()
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

        # Layout that can reflow (row + color picker)
        self.row_and_picker = QBoxLayout(hbox_dir_left_to_right())
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

        # Color picker panel
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

        # ---------------- Right side ----------------
        self.right = QWidget()
        self.right_lay = QVBoxLayout(self.right)
        self.right_lay.setContentsMargins(0, 0, 0, 0)
        self.right_lay.setSpacing(10)

        # Simulation
        self.sim_panel = Panel("Light Simulation")
        self.right_lay.addWidget(self.sim_panel, 3)
        self.sim = SimulationWidget()
        self.sim_panel.body.addWidget(self.sim, 1)

        # Preset under simulation
        self.preset_panel = Panel("Preset")
        self.right_lay.addWidget(self.preset_panel, 0)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)
        preset_row.addWidget(QLabel("Name:"))

        self.preset_name = QLineEdit(self.state.preset_name)
        self.preset_name.setPlaceholderText("Preset name (for SAVE/LOAD)")
        preset_row.addWidget(self.preset_name, 1)
        self.preset_panel.body.addLayout(preset_row)

        preset_btns = QHBoxLayout()
        preset_btns.setSpacing(10)
        self.btn_save = QPushButton("SAVE")
        self.btn_load = QPushButton("LOAD")
        for b in [self.btn_save, self.btn_load]:
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preset_btns.addWidget(self.btn_save)
        preset_btns.addWidget(self.btn_load)
        self.preset_panel.body.addLayout(preset_btns)

        # Audio analysis / visualizer
        self.audio_panel = Panel("Audio Analysis")
        self.right_lay.addWidget(self.audio_panel, 2)

        bpm_row = QHBoxLayout()
        bpm_row.setSpacing(10)
        self.bpm_lbl = QLabel("BPM: --")
        self.bpm_lbl.setObjectName("Subtle")
        bpm_row.addWidget(self.bpm_lbl)
        bpm_row.addStretch(1)
        self.audio_panel.body.addLayout(bpm_row)

        self.visualizer = SimpleVisualizer(nbars=64)
        self.audio_panel.body.addWidget(self.visualizer, 1)

        # Put both panes into splitter
        self.split.addWidget(self.left)
        self.split.addWidget(self.right)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)

        # ============================================================
        # Wiring
        # ============================================================
        for s in [c.slider for c in self.cards]:
            s.valueChanged.connect(self._on_controls_changed)
        self.sw_auto.toggled.connect(self._on_controls_changed)
        self.sw_auto_move.toggled.connect(self._on_controls_changed)
        self.preset_name.textChanged.connect(self._on_controls_changed)

        self.btn_save.clicked.connect(self.save_clicked.emit)
        self.btn_load.clicked.connect(self.load_clicked.emit)

        self.btn_import.clicked.connect(self.import_clicked.emit)
        self.btn_play.clicked.connect(self.play_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_replay.clicked.connect(self.replay_clicked.emit)

        self.btn_record.clicked.connect(self._start_recording)
        self.btn_stoprec.clicked.connect(self._stop_recording)

        # Button groups for responsive sizing
        self._transport_buttons = [
            self.btn_play,
            self.btn_stop,
            self.btn_replay,
            self.btn_import,
            self.btn_record,
            self.btn_stoprec,
        ]
        self._all_main_buttons = self._transport_buttons + [self.btn_save, self.btn_load]

        # Initial states
        self.btn_stoprec.setEnabled(False)
        self._set_swatch_color(self.card_r.slider.value(), self.card_g.slider.value(), self.card_b.slider.value())
        self._on_controls_changed()

        QTimer.singleShot(0, self._apply_responsive)

    # ============================================================
    # Responsive layout
    # ============================================================
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_responsive()

    def _apply_responsive(self):
        w = max(1, self.width())
        h = max(1, self.height())
        sw = w / float(self._base_w)
        sh = h / float(self._base_h)
        s = float(np.clip(min(sw, sh), 0.58, 1.0))
        self._last_scale = s

        # Scale panels (if Panel supports it)
        for p in [
            self.timeline_panel,
            self.transport_panel,
            self.fx_panel,
            self.slider_panel,
            self.sim_panel,
            self.preset_panel,
            self.audio_panel,
            self.color_panel,
        ]:
            try:
                p.apply_scale(s)
            except Exception:
                pass

        # Button sizing
        btn_h = int(max(34, round(48 * s)))
        fx_h = int(max(30, round(56 * s)))
        for b in self._all_main_buttons:
            b.setMinimumHeight(btn_h)
        for b in self.fx_buttons.values():
            b.setMinimumHeight(fx_h)

        # Fonts
        self.time_lbl.setFont(QFont("Segoe UI", int(max(11, round(14 * s))), 800))
        for lb in [self.status_lbl, self.bpm_lbl, self.wav_lbl, self.time_cur, self.time_tot, self.color_hex] + self.switch_labels:
            f = lb.font()
            f.setPointSizeF(max(8.0, 10.0 * s))
            lb.setFont(f)

        for swt in self.switches:
            try:
                swt.apply_scale(s)
            except Exception:
                pass

        try:
            self.visualizer.apply_scale(s)
        except Exception:
            pass
        try:
            self.sim.apply_scale(s)
        except Exception:
            pass

        self.scrub.setFixedHeight(int(max(18, round(22 * s))))
        self.color_swatch.setMinimumHeight(int(max(56, round(86 * s))))

        # Reflow slider row and color picker when narrow
        narrow = self.width() < 1120
        self.row_and_picker.setDirection(hbox_dir_top_to_bottom() if narrow else hbox_dir_left_to_right())

        if narrow:
            self.color_panel.setMaximumWidth(16777215)
        else:
            self.color_panel.setMaximumWidth(int(max(150, round(240 * s))))

        self._apply_slider_row_geometry(s, narrow)

        # Keep transport panel visually aligned with timeline panel
        try:
            self.transport_panel.setFixedHeight(self.timeline_panel.sizeHint().height())
        except Exception:
            pass

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
            try:
                c.apply_size(card_w, card_h, slider_h, s)
            except Exception:
                pass

    # ============================================================
    # Player / timeline
    # ============================================================
    def _scrub_start(self):
        self._user_scrubbing = True

    def _scrub_commit(self):
        self._user_scrubbing = False
        ratio = float(self.scrub.value()) / 1000.0
        self.seek_ratio.emit(ratio)

    def set_player_position(self, cur_sec: float, total_sec: float):
        self._cur_sec = float(cur_sec)

        self.time_lbl.setText(fmt_time(cur_sec))
        self.time_cur.setText(fmt_time(cur_sec))
        self.time_tot.setText(fmt_time(total_sec))

        self.timeline.set_position(cur_sec, total_sec)

        if not self._user_scrubbing and total_sec > 0.0:
            ratio = float(np.clip(cur_sec / total_sec, 0.0, 1.0))
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(ratio * 1000))
            self.scrub.blockSignals(False)

    def set_beats(self, bpm: float, beat_times: list[float]):
        # Beat grid shown on timeline
        self.timeline.set_beats(float(bpm or 0.0), beat_times or [])
        if bpm and bpm > 0:
            self.bpm_lbl.setText(f"BPM: {bpm:.1f}")

    # ============================================================
    # Color
    # ============================================================
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
        from ..qt_compat import QColor

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

    # ============================================================
    # State / controls
    # ============================================================
    def _on_fx_clicked(self, btn: QToolButton, checked: bool):
        if checked:
            # enforce single active FX visually
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

        self._on_controls_changed()

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

        # Update numeric labels on slider cards
        self.card_dim.set_value_text(self.state.intensity)
        self.card_r.set_value_text(self.state.r)
        self.card_g.set_value_text(self.state.g)
        self.card_b.set_value_text(self.state.b)
        self.card_st.set_value_text(self.state.strobe_rate)
        self.card_pan.set_value_text(self.state.pan)
        self.card_tilt.set_value_text(self.state.tilt)

        # Lock pan/tilt when auto_move is active
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

    # ============================================================
    # Audio + visualizer
    # ============================================================
    def set_loaded_file(self, path: str):
        self.wav_lbl.setText(f"WAV: {path}")

    def update_visualizer(self, feat: AudioFeatures):
        self.visualizer.update_from_spectrum(feat.spectrum)
        self.bpm_lbl.setText(f"BPM: {feat.bpm:0.0f}" if (feat.bpm and feat.bpm > 0.0) else "BPM: --")

    # ============================================================
    # Presets (controls + timeline cues)
    # ============================================================
    def export_preset(self) -> dict:
        return {
            "preset_name": self.preset_name.text().strip() or "Untitled",
            "controls": {
                "fx": self.state.fx,
                "auto_audio": self.sw_auto.isChecked(),
                "auto_move": self.sw_auto_move.isChecked(),
                "intensity": int(self.card_dim.slider.value()),
                "r": int(self.card_r.slider.value()),
                "g": int(self.card_g.slider.value()),
                "b": int(self.card_b.slider.value()),
                "strobe_rate": int(self.card_st.slider.value()),
                "pan": int(self.card_pan.slider.value()),
                "tilt": int(self.card_tilt.slider.value()),
            },
            "cues": self.timeline.export_cues(),
        }

    def import_preset(self, data: dict):
        try:
            self.preset_name.setText(str(data.get("preset_name", "Untitled")))
            c = data.get("controls", {}) or {}

            # Sliders first
            self.card_dim.slider.setValue(int(c.get("intensity", self.card_dim.slider.value())))
            self.card_r.slider.setValue(int(c.get("r", self.card_r.slider.value())))
            self.card_g.slider.setValue(int(c.get("g", self.card_g.slider.value())))
            self.card_b.slider.setValue(int(c.get("b", self.card_b.slider.value())))
            self.card_st.slider.setValue(int(c.get("strobe_rate", self.card_st.slider.value())))
            self.card_pan.slider.setValue(int(c.get("pan", self.card_pan.slider.value())))
            self.card_tilt.slider.setValue(int(c.get("tilt", self.card_tilt.slider.value())))

            self.sw_auto.setChecked(bool(c.get("auto_audio", self.sw_auto.isChecked())))
            self.sw_auto_move.setChecked(bool(c.get("auto_move", self.sw_auto_move.isChecked())))

            # FX last (reflect in buttons)
            fx = str(c.get("fx", "") or "")
            if fx:
                for name, btn in self.fx_buttons.items():
                    btn.blockSignals(True)
                    btn.setChecked(name.lower() == fx.lower())
                    btn.blockSignals(False)
                self.state.fx = fx
                # Track active button
                self._active_fx_btn = None
                for btn in self.fx_buttons.values():
                    if btn.isChecked():
                        self._active_fx_btn = btn
                        break
            else:
                for btn in self.fx_buttons.values():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
                self.state.fx = ""
                self._active_fx_btn = None

            self.timeline.set_cues(list(data.get("cues", []) or []))
            self._on_controls_changed()
        except Exception as e:
            QMessageBox.warning(self, "Load preset", f"Preset file invalid: {e}")

    # ============================================================
    # Recording UI state (timeline cue recording handled elsewhere)
    # ============================================================
    def _start_recording(self):
        self._recording = True
        self.status_lbl.setText("Recording")
        self.btn_record.setEnabled(False)
        self.btn_stoprec.setEnabled(True)

    def _stop_recording(self):
        self._recording = False
        self.status_lbl.setText("Stopped")
        self.btn_record.setEnabled(True)
        self.btn_stoprec.setEnabled(False)
