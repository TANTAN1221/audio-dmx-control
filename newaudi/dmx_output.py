# dmx_output.py
from __future__ import annotations
from dataclasses import dataclass
import socket
import time
from typing import List

@dataclass
class OutputConfig:
    enabled: bool = False
    blackout: bool = False
    protocol: str = "off"          # "off", "artnet", "sacn"
    target_ip: str = "255.255.255.255"
    universe: int = 0              # Art-Net universe (0-based)
    sacn_universe: int = 1         # sACN universe (1-based)
    start_address: int = 1         # 1..512
    fps: int = 30

class OutputEngine:
    def __init__(self):
        self.cfg = OutputConfig()
        self.dmx = bytearray(512)
        self._last_send = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def apply_config(self, cfg: OutputConfig):
        self.cfg = cfg

    def set_channels_from_faders(self, faders: List[int]):
        """
        Map faders 0..100 into DMX 0..255 starting at start_address.
        """
        start = max(1, min(512, int(self.cfg.start_address))) - 1
        for i, v in enumerate(faders):
            ch = start + i
            if 0 <= ch < 512:
                self.dmx[ch] = max(0, min(255, int((v / 100.0) * 255)))

    def tick(self):
        if not self.cfg.enabled or self.cfg.protocol == "off":
            return

        now = time.time()
        interval = 1.0 / max(1, int(self.cfg.fps))
        if now - self._last_send < interval:
            return
        self._last_send = now

        payload = bytes(512) if self.cfg.blackout else bytes(self.dmx)

        # NOTE: This is a placeholder sender.
        # Real Art-Net/sACN framing is more involved.
        # Keeping it minimal so your UI + audio works without extra deps.
        self._sock.sendto(payload, (self.cfg.target_ip, 6454))  # 6454 is Art-Net port

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass
