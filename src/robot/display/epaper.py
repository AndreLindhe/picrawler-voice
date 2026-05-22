from __future__ import annotations

"""
E-paper display driver for Waveshare 2.13" V4 (SSD1680, 250×122 px).

Wiring (Robot HAT):
  VCC  → 3V3      GND  → GND
  DIN  → BCM 10   CLK  → BCM 11
  CS   → BCM  8   DC   → BCM  4  (D1)
  RST  → BCM 17   BUSY → BCM 24  (not used in software — fixed delays instead)

Uses raw lgpio + spidev to avoid BUSY-pin timing issues with the waveshare
library.  All refreshes are full-update; partial updates are skipped because
BUSY is not reliably connected.

Subscribes to bus events and redraws when status changes, throttled to at
most one refresh per _MIN_REFRESH_S seconds.
"""

import asyncio
import logging
import socket
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.bus import MessageBus
    from ..core.state import StateManager

from ..core.events import (
    BatteryCritical,
    BatteryLow,
    BehaviorEnded,
    BehaviorStarted,
    ObstacleCleared,
    ObstacleDetected,
    SpeakRequest,
    Transcript,
    WakeWordDetected,
)

logger = logging.getLogger(__name__)

_DC_PIN  = 4    # D1 on Robot HAT
_RST_PIN = 17   # D0 on Robot HAT
_SPI_BUS = 0
_SPI_DEV = 0
_SPI_HZ  = 2_000_000

_MIN_REFRESH_S = 6.0   # e-ink full refresh takes ~4s; don't queue faster than this


def _get_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "no network"
    finally:
        s.close()


class EpaperDisplay:
    """
    Async display task. Redraws the screen whenever robot status changes.

    Subscribes to: BatteryLow, BatteryCritical, BehaviorStarted,
    BehaviorEnded, ObstacleDetected, ObstacleCleared, WakeWordDetected,
    Transcript, SpeakRequest.
    """

    def __init__(self, bus: "MessageBus", state: "StateManager") -> None:
        self._bus   = bus
        self._state = state

        self._battery_pct: int          = 0
        self._battery_v:   float        = 0.0
        self._behavior:    str          = "Starting"
        self._obstacle:    bool         = False
        self._sonar_cm:    Optional[float] = None
        self._listening:   bool         = False
        self._speaking:    bool         = False

        self._h   = None   # lgpio chip handle
        self._spi = None   # spidev handle
        self._font_large = None
        self._font_med   = None
        self._font_small = None
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        try:
            await asyncio.to_thread(self._init_display)
        except Exception:
            logger.exception("display: init failed — display task exiting")
            return

        await self._draw()

        topics = [
            BatteryLow.topic, BatteryCritical.topic,
            BehaviorStarted.topic, BehaviorEnded.topic,
            ObstacleDetected.topic, ObstacleCleared.topic,
            WakeWordDetected.topic, Transcript.topic,
            SpeakRequest.topic,
        ]
        async with self._bus.subscribe(*topics) as q:
            while True:
                event = await q.get()
                self._handle_event(event)
                # throttle: skip if a refresh just happened
                if time.monotonic() - self._last_refresh >= _MIN_REFRESH_S:
                    await self._draw()

    def _handle_event(self, event) -> None:
        if isinstance(event, (BatteryLow, BatteryCritical)):
            self._battery_pct = event.percent
            self._battery_v   = event.voltage
        elif isinstance(event, BehaviorStarted):
            self._behavior = event.name
            self._listening = False
            self._speaking  = False
        elif isinstance(event, BehaviorEnded):
            self._behavior = "Idle"
        elif isinstance(event, ObstacleDetected):
            self._obstacle  = True
            self._sonar_cm  = event.distance_cm
        elif isinstance(event, ObstacleCleared):
            self._obstacle  = False
            self._sonar_cm  = event.distance_cm
        elif isinstance(event, WakeWordDetected):
            self._listening = True
        elif isinstance(event, Transcript):
            self._listening = False
        elif isinstance(event, SpeakRequest):
            self._speaking  = True

    # ------------------------------------------------------------------
    # Hardware init
    # ------------------------------------------------------------------

    def _init_display(self) -> None:
        import lgpio, spidev
        from PIL import ImageFont

        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, _DC_PIN)
        lgpio.gpio_claim_output(self._h, _RST_PIN)

        self._spi = spidev.SpiDev()
        self._spi.open(_SPI_BUS, _SPI_DEV)
        self._spi.max_speed_hz = _SPI_HZ
        self._spi.mode = 0

        try:
            fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            self._font_large = ImageFont.truetype(fp, 20)
            self._font_med   = ImageFont.truetype(fp, 14)
            self._font_small = ImageFont.truetype(fp, 11)
        except Exception:
            self._font_large = ImageFont.load_default()
            self._font_med   = self._font_large
            self._font_small = self._font_large

        self._epd_init()
        logger.info("display: initialised (250×122 landscape, DC=BCM%d)", _DC_PIN)

    def _dc(self, v: int) -> None:
        import lgpio
        lgpio.gpio_write(self._h, _DC_PIN, v)

    def _rst(self, v: int) -> None:
        import lgpio
        lgpio.gpio_write(self._h, _RST_PIN, v)

    def _cmd(self, b: int) -> None:
        self._dc(0); self._spi.writebytes([b])

    def _data(self, b: int) -> None:
        self._dc(1); self._spi.writebytes([b])

    def _data_buf(self, buf) -> None:
        self._dc(1); self._spi.writebytes2(buf)

    def _epd_init(self) -> None:
        """SSD1680 full initialisation sequence."""
        self._rst(1); time.sleep(0.2)
        self._rst(0); time.sleep(0.1)
        self._rst(1); time.sleep(1.0)

        self._cmd(0x12); time.sleep(0.5)           # SWRESET

        self._cmd(0x01)                             # driver output control
        self._data(0xF9); self._data(0x00); self._data(0x00)

        self._cmd(0x11); self._data(0x03)           # data entry: X+, Y+

        self._cmd(0x44); self._data(0x00); self._data(0x0F)   # X: bytes 0–15
        self._cmd(0x45)                             # Y: rows 0–249
        self._data(0x00); self._data(0x00)
        self._data(0xF9); self._data(0x00)

        self._cmd(0x3C); self._data(0x05)           # border waveform
        self._cmd(0x21); self._data(0x00); self._data(0x80)   # display update ctrl
        self._cmd(0x18); self._data(0x80)           # internal temperature sensor

        self._cmd(0x4E); self._data(0x00)           # cursor X = 0
        self._cmd(0x4F); self._data(0x00); self._data(0x00)   # cursor Y = 0
        time.sleep(0.1)

    def _epd_full_refresh(self, buf) -> None:
        """Write buffer and trigger full display update (~4s)."""
        # Reset cursor before writing
        self._cmd(0x4E); self._data(0x00)
        self._cmd(0x4F); self._data(0x00); self._data(0x00)

        self._cmd(0x24)
        self._data_buf(buf)

        self._cmd(0x22); self._data(0xF7); self._cmd(0x20)
        time.sleep(5.0)   # wait for e-ink to fully settle

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    async def _draw(self) -> None:
        try:
            await asyncio.to_thread(self._draw_sync)
        except Exception:
            logger.exception("display: draw failed")

    def _draw_sync(self) -> None:
        from PIL import Image, ImageDraw

        # Draw in landscape (250 wide × 122 tall)
        W, H = 250, 122
        img = Image.new("1", (W, H), 255)
        d   = ImageDraw.Draw(img)

        # ── Battery bar ───────────────────────────────────────────────
        bat_text = f"BAT: {self._battery_pct}%  {self._battery_v:.1f}V"
        d.text((4, 4), bat_text, font=self._font_med, fill=0)
        bx, by, bw, bh = 162, 5, 80, 13
        d.rectangle([bx, by, bx + bw, by + bh], outline=0)
        fw = int(bw * max(0, min(100, self._battery_pct)) / 100)
        if fw:
            d.rectangle([bx + 1, by + 1, bx + fw, by + bh - 1], fill=0)

        d.line([(0, 22), (W, 22)], fill=0, width=1)

        # ── Status ────────────────────────────────────────────────────
        if self._obstacle:
            status = f"OBSTACLE  {self._sonar_cm:.0f}cm" if self._sonar_cm else "OBSTACLE"
        elif self._listening:
            status = "Listening..."
        elif self._speaking:
            status = "Speaking"
        else:
            status = self._behavior
        d.text((4, 26), status, font=self._font_large, fill=0)

        d.line([(0, 52), (W, 52)], fill=0, width=1)

        # ── Sonar ─────────────────────────────────────────────────────
        sonar_txt = f"Sonar: {self._sonar_cm:.0f} cm" if self._sonar_cm else "Sonar: --"
        d.text((4, 56), sonar_txt, font=self._font_med, fill=0)

        d.line([(0, 74), (W, 74)], fill=0, width=1)

        # ── IP + time ─────────────────────────────────────────────────
        ip = _get_ip()
        d.text((4, 78), f"IP: {ip}", font=self._font_small, fill=0)
        ts = time.strftime("%H:%M  %d %b")
        d.text((4, 98), ts, font=self._font_small, fill=0)

        # Rotate 270° (= 90° CW): landscape → column-scanned portrait buffer
        img_r = img.rotate(270, expand=True)
        buf   = bytearray(img_r.convert("1").tobytes("raw"))

        self._epd_full_refresh(buf)
        self._last_refresh = time.monotonic()
