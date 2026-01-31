from __future__ import annotations

def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    mm = int(sec // 60)
    ss = int(sec % 60)
    return f"{mm:02d}:{ss:02d}"
