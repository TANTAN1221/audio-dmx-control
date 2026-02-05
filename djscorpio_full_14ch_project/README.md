# DJScorpio Moving Head Controller (UI + Audio + FX) + USB DMX (14CH)

This build maps the UI to the DJScorpio 14CH manual you provided.

## Install
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run
```powershell
.\.venv\Scripts\python.exe app.py
```

## Fixture setup
- Mode: 14CH
- Address: 001

## USB DMX dongle
- Change `com_port="COM3"` in app.py if needed (Device Manager -> Ports).
- Toggle **DMX Output** ON in the UI.

## Channels (14CH)
CH1 Pan, CH2 Pan fine, CH3 Tilt, CH4 Tilt fine, CH5 motor speed, CH6 dimmer,
CH7 strobe, CH8 color wheel, CH9 pattern/gobo, CH10 prism, CH11 programs,
CH12 reset (disabled), CH13 strip effect, CH14 strip speed.

NOTE: This fixture uses a **color wheel** on CH8 (not RGB mixing). The app maps your RGB picker to the nearest wheel slot.
