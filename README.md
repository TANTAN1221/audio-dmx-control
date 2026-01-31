# Lighting FX + Audio Visualizer (Modular)

Run:
```bash
python -m pip install PyQt6 numpy sounddevice soundfile
python main.py
```

Files:
- `dj_lighting/ui/` GUI widgets (panels, sliders, visualizer, simulation)
- `dj_lighting/audio/` audio feature extraction + WAV worker
- `dj_lighting/lighting/` FX engine / control state
- `dj_lighting/output/` DMX output (Art-Net placeholder)

Window behavior:
- The main window is forced to **stay maximized** and fixed to the screen's available area.
