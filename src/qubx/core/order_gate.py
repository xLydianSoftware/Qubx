"""Pluggable pre-trade order gates for Qubx (software risk gate + Proteus hardware adapter)."""

import json
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from qubx import logger
from qubx.core.basics import Order, OrderRequest, resolve_reduce_only

PROTEUS_CLK_HZ = 60_000_000
PROTEUS_NS_PER_CLK = 1e9 / PROTEUS_CLK_HZ

OP_NEW_ORDER = 0x4F
OP_CANCEL = 0x58
OP_HEARTBEAT = 0x48

PROTEUS_AUDIT_CODES: dict[int, tuple[bool, str]] = {
    0xA0: (True, "PASS_NEW_ORDER"),
    0xA1: (True, "PASS_CANCEL"),
    0xA2: (True, "PASS_HEARTBEAT"),
    0xE1: (False, "RATE_LIMIT"),
    0xE2: (False, "KILL_SWITCH"),
    0xE3: (False, "BAD_OPCODE"),
    0xE4: (False, "FAT_FINGER"),
}


@dataclass(frozen=True)
class OrderGateDecision:
    """Outcome of a pre-trade order or cancel gate check."""

    allowed: bool
    code: str
    reason: str = ""
    latency_ns: float | None = None


@runtime_checkable
class IOrderGate(Protocol):
    """Pre-trade risk boundary evaluated synchronously before venue submission."""

    def check_order(self, request: OrderRequest) -> OrderGateDecision: ...

    def check_cancel(self, order: Order) -> OrderGateDecision: ...

    def set_kill_switch(self, armed: bool) -> None: ...


class SoftwareOrderGate:
    """In-process pre-trade risk gate: sliding-window rate limiter, fat-finger limits, and kill-switch.

    Cancels never consume rate-limit tokens and bypass rate-limit / fat-finger checks so risk-reducing
    order pulls stay available during bursts.
    """

    def __init__(
        self,
        max_orders: int | None = None,
        window_seconds: float = 1.0,
        max_order_qty: float | None = None,
        max_order_notional: float | None = None,
        allow_reduce_only_when_throttled: bool = True,
        block_cancels_on_kill: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_orders is not None and max_orders <= 0:
            raise ValueError("max_orders must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._max_orders = max_orders
        self._window_seconds = window_seconds
        self._max_order_qty = max_order_qty
        self._max_order_notional = max_order_notional
        self._allow_reduce_only_when_throttled = allow_reduce_only_when_throttled
        self._block_cancels_on_kill = block_cancels_on_kill
        self._clock = clock or time.monotonic
        self._timestamps: deque[float] = deque()
        self._armed = True

    @property
    def is_armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        self._armed = True

    def trip(self) -> None:
        self._armed = False

    def set_kill_switch(self, armed: bool) -> None:
        self._armed = bool(armed)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def check_order(self, request: OrderRequest) -> OrderGateDecision:
        if not self._armed:
            return OrderGateDecision(False, "KILL_SWITCH", f"kill-switch active for {request.instrument.symbol}")

        qty = abs(request.quantity)
        if self._max_order_qty is not None and qty > self._max_order_qty:
            return OrderGateDecision(
                False,
                "FAT_FINGER",
                f"{request.instrument.symbol} qty {qty} exceeds max_order_qty {self._max_order_qty}",
            )

        if self._max_order_notional is not None and request.price is not None:
            notional = qty * abs(request.price) * request.instrument.quantity_multiplier
            if notional > self._max_order_notional:
                return OrderGateDecision(
                    False,
                    "FAT_FINGER",
                    f"{request.instrument.symbol} notional {notional:.2f} exceeds max {self._max_order_notional:.2f}",
                )

        if self._max_orders is not None:
            is_reduce_only = bool(resolve_reduce_only(request.options or {}))
            now = self._clock()
            self._prune(now)
            if len(self._timestamps) >= self._max_orders:
                if not (self._allow_reduce_only_when_throttled and is_reduce_only):
                    return OrderGateDecision(
                        False,
                        "RATE_LIMIT",
                        f"rate limit {self._max_orders}/{self._window_seconds}s exceeded for {request.instrument.symbol}",
                    )
            else:
                self._timestamps.append(now)

        return OrderGateDecision(True, "PASS")

    def check_cancel(self, order: Order) -> OrderGateDecision:
        if not self._armed and self._block_cancels_on_kill:
            return OrderGateDecision(False, "KILL_SWITCH", f"kill-switch blocks cancel for {order.client_order_id}")
        return OrderGateDecision(True, "PASS_CANCEL")


class ProteusOrderGate:
    """Hardware-in-the-loop order gate delegating pre-trade checks to Proteus RTL v2 silicon.

    Talks to either a local Verilator runner (`proteus/examples/qubx_risk_gate/build/sim_risk_gate`)
    or a live Tang Nano 9K + Raspberry Pi Pico 2 bridge over USB-serial using the shared
    newline-delimited JSON protocol.
    """

    def __init__(
        self,
        image_hex: str | None = None,
        sim_bin: str | Path | None = None,
        serial_port: str | None = None,
        serial_baud: int = 115200,
        symbol_ids: dict[str, int] | None = None,
        qty_scale: float = 1.0,
        transport: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._symbol_ids = dict(symbol_ids or {})
        self._qty_scale = qty_scale
        self._proc: subprocess.Popen[str] | None = None
        self._ser: Any = None
        self._transport = transport
        self._armed = True

        if transport is None:
            if serial_port is not None:
                try:
                    import serial  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise RuntimeError("pyserial is required for ProteusOrderGate serial_port mode") from exc
                self._ser = serial.Serial(serial_port, serial_baud, timeout=2.0)
            elif sim_bin is not None:
                path = Path(sim_bin)
                if not path.exists():
                    raise FileNotFoundError(f"Proteus Verilator simulator not found: {path}")
                self._proc = subprocess.Popen(
                    [str(path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            else:
                raise ValueError("ProteusOrderGate requires sim_bin, serial_port, or transport")

        if image_hex is not None:
            resp = self._cmd(f"LOAD {image_hex}")
            if not resp.get("ok") or not resp.get("running"):
                raise RuntimeError(f"Failed to load Proteus risk-gate table: {resp}")

    def _cmd(self, line: str) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(line)
        if self._ser is not None:
            self._ser.write((line + "\n").encode("ascii"))
            self._ser.flush()
            out = self._ser.readline().decode("utf-8", errors="replace")
            if not out:
                raise RuntimeError("Proteus serial bridge timed out")
            return json.loads(out)
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline()
        if not out:
            raise RuntimeError("Proteus Verilator simulator exited unexpectedly")
        return json.loads(out)

    def _sym_id(self, symbol: str) -> int:
        if symbol not in self._symbol_ids:
            self._symbol_ids[symbol] = (len(self._symbol_ids) % 254) + 1
        return self._symbol_ids[symbol]

    def _encode_frame(self, opcode: int, symbol: str, price: float | None, qty: float) -> bytes:
        sym_b = self._sym_id(symbol) & 0xFF
        px_b = int(round(price)) & 0xFF if price is not None else 128
        qty_b = max(0, min(255, int(round(abs(qty) * self._qty_scale))))
        return bytes([opcode & 0xFF, sym_b, px_b, qty_b])

    def _evaluate(self, frame: bytes, label: str) -> OrderGateDecision:
        resp = self._cmd(f"SEND_FRAME {frame.hex()} -1 0")
        raw_tx = bytes.fromhex(resp.get("tx_hex", ""))
        audits = list(bytes.fromhex(resp.get("audits_hex", "")))
        lat_clks = int(resp.get("lat_clks", -1))
        lat_ns = None if lat_clks < 0 else lat_clks * PROTEUS_NS_PER_CLK

        passed = raw_tx == frame
        code = "PASS" if passed else "HARDWARE_DROP"
        if audits:
            _, code = PROTEUS_AUDIT_CODES.get(audits[-1], (passed, f"HW_0x{audits[-1]:02X}"))
        reason = "" if passed else f"Proteus silicon rejected {label} ({code})"
        return OrderGateDecision(allowed=passed, code=code, reason=reason, latency_ns=lat_ns)

    def check_order(self, request: OrderRequest) -> OrderGateDecision:
        frame = self._encode_frame(OP_NEW_ORDER, request.instrument.symbol, request.price, request.quantity)
        return self._evaluate(frame, f"order {request.instrument.symbol} qty={request.quantity}")

    def check_cancel(self, order: Order) -> OrderGateDecision:
        frame = self._encode_frame(OP_CANCEL, order.instrument.symbol, order.price, order.quantity)
        return self._evaluate(frame, f"cancel {order.client_order_id}")

    def set_kill_switch(self, armed: bool) -> None:
        self._armed = bool(armed)
        self._cmd(f"SET_KILL {1 if self._armed else 0}")

    def advance_us(self, us: float) -> None:
        clks = max(1, int(round(us * PROTEUS_CLK_HZ / 1e6)))
        self._cmd(f"ADVANCE {clks}")

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception as exc:
                logger.debug(f"[ProteusOrderGate] :: Error closing serial port: {exc}")
            self._ser = None
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._cmd("QUIT")
                except Exception:
                    pass
                self._proc.terminate()
                self._proc.wait(timeout=3)
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            self._proc = None

    def __enter__(self) -> "ProteusOrderGate":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
