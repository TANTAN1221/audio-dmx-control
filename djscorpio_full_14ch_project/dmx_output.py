# dmx_output.py  (UNCHANGED)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time
import socket

try:
    import serial  # only needed for usbpro
except Exception:
    serial = None


@dataclass
class OutputConfig:
    enabled: bool = False
    blackout: bool = False

    # "off" | "usbpro" | "artnet"
    protocol: str = "off"

    # USB Pro settings
    com_port: str = "COM3"
    baudrate: int = 57600

    # Art-Net settings
    target_ip: str = "127.0.0.1"
    universe: int = 0  # Art-Net "Net/SubUni"

    # DMX addressing + send rate
    start_address: int = 1
    fps: int = 30


class _UsbDmxProSender:
    START = 0x7E
    END = 0xE7
    LABEL_SEND_DMX = 0x06

    def __init__(self, port: str, baudrate: int):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def send_512(self, dmx512: bytes):
        if len(dmx512) != 512:
            raise ValueError("dmx512 must be 512 bytes")
        payload = b"\x00" + dmx512
        length = len(payload)
        pkt = bytearray([self.START, self.LABEL_SEND_DMX, length & 0xFF, (length >> 8) & 0xFF])
        pkt += payload
        pkt.append(self.END)
        self.ser.write(pkt)


class _ArtNetSender:
    PORT = 6454

    def __init__(self, target_ip: str):
        self.target_ip = target_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    @staticmethod
    def _make_artdmx_packet(universe: int, dmx512: bytes, seq: int = 0) -> bytes:
        header = bytearray()
        header += b"Art-Net\x00"
        header += (0x5000).to_bytes(2, "little")
        header += (14).to_bytes(2, "big")
        header += bytes([seq & 0xFF])
        header += bytes([0])  # physical
        header += int(universe & 0xFFFF).to_bytes(2, "little")
        length = len(dmx512)
        header += int(length & 0xFFFF).to_bytes(2, "big")
        return bytes(header) + dmx512

    def send_512(self, universe: int, dmx512: bytes, seq: int = 0):
        if len(dmx512) != 512:
            raise ValueError("dmx512 must be 512 bytes")
        pkt = self._make_artdmx_packet(universe, dmx512, seq=seq)
        self.sock.sendto(pkt, (self.target_ip, self.PORT))


class OutputEngine:
    def __init__(self):
        self.cfg = OutputConfig()
        self.dmx = bytearray(512)
        self._last_send = 0.0

        self._usb: Optional[_UsbDmxProSender] = None
        self._art: Optional[_ArtNetSender] = None

        self._open_port: Optional[str] = None
        self._open_baud: Optional[int] = None
        self._open_ip: Optional[str] = None

        self._seq = 0

    def apply_config(self, cfg: OutputConfig):
        reopen_usb = (
            self.cfg.protocol != cfg.protocol or
            self.cfg.com_port != cfg.com_port or
            self.cfg.baudrate != cfg.baudrate
        )
        reopen_art = (
            self.cfg.protocol != cfg.protocol or
            self.cfg.target_ip != cfg.target_ip
        )

        self.cfg = cfg

        if reopen_usb:
            self._close_usb()
        if reopen_art:
            self._close_art()

    def _ensure_usb(self):
        if not self.cfg.enabled or self.cfg.protocol != "usbpro":
            return
        if self._usb is None or self._open_port != self.cfg.com_port or self._open_baud != self.cfg.baudrate:
            self._close_usb()
            self._usb = _UsbDmxProSender(self.cfg.com_port, self.cfg.baudrate)
            self._open_port = self.cfg.com_port
            self._open_baud = self.cfg.baudrate

    def _ensure_artnet(self):
        if not self.cfg.enabled or self.cfg.protocol != "artnet":
            return
        if self._art is None or self._open_ip != self.cfg.target_ip:
            self._close_art()
            self._art = _ArtNetSender(self.cfg.target_ip)
            self._open_ip = self.cfg.target_ip

    def _close_usb(self):
        if self._usb is not None:
            try:
                self._usb.close()
            finally:
                self._usb = None
                self._open_port = None
                self._open_baud = None

    def _close_art(self):
        if self._art is not None:
            try:
                self._art.close()
            finally:
                self._art = None
                self._open_ip = None

    def set_channels_from_values_0_255(self, values: List[int]):
        start = max(1, min(512, int(self.cfg.start_address))) - 1
        for i, v in enumerate(values):
            ch = start + i
            if 0 <= ch < 512:
                self.dmx[ch] = max(0, min(255, int(v)))

    def tick(self):
        if not self.cfg.enabled or self.cfg.protocol == "off":
            return

        now = time.time()
        interval = 1.0 / max(1, int(self.cfg.fps))
        if now - self._last_send < interval:
            return
        self._last_send = now

        payload = bytes(512) if self.cfg.blackout else bytes(self.dmx)

        if self.cfg.protocol == "usbpro":
            self._ensure_usb()
            if self._usb is not None:
                self._usb.send_512(payload)

        elif self.cfg.protocol == "artnet":
            self._ensure_artnet()
            if self._art is not None:
                self._seq = (self._seq + 1) & 0xFF
                self._art.send_512(self.cfg.universe, payload, seq=self._seq)

    def close(self):
        self._close_usb()
        self._close_art()
