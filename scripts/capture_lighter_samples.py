"""
Script to capture live WebSocket samples from Lighter exchange.

This script connects to Lighter mainnet WebSocket, subscribes to various
channels, and captures sample messages for testing purposes.

Usage:
    poetry run python scripts/capture_lighter_samples.py

Captures:
    - Order book snapshots and updates (BTC-USDC, ETH-USDC)
    - Trade messages
    - Market stats
    - Connection/subscription confirmations

Output:
    Saves samples to: tests/qubx/connectors/xlighter/test_data/samples/
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from qubx.connectors.xlighter import LighterWebSocketManager


class LighterSampleCapture:
    """Capture Lighter WebSocket samples"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for captured samples
        self.samples = {
            "orderbook": [],
            "trades": [],
            "market_stats": [],
            "system": [],
        }

        # Statistics
        self.message_count = 0
        self.start_time = None

    async def capture_handler(self, message: dict, message_type: str):
        """Handler for captured messages"""
        self.message_count += 1

        # Add metadata
        message_with_meta = {
            "captured_at": datetime.now().isoformat(),
            "message_number": self.message_count,
            "message_type": message_type,
            "data": message,
        }

        # Store in appropriate category
        if message_type in self.samples:
            self.samples[message_type].append(message_with_meta)
        else:
            self.samples["system"].append(message_with_meta)

        # Log progress
        msg_type = message.get("type", "unknown")
        channel = message.get("channel", "unknown")
        print(f"[{self.message_count:3d}] {msg_type:<30} | {channel:<20} | {message_type}")

    async def run(self, duration_seconds: int = 30):
        """
        Run capture for specified duration.

        Args:
            duration_seconds: How long to capture messages
        """
        print(f"\n{'='*70}")
        print(f"Lighter WebSocket Sample Capture")
        print(f"{'='*70}")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        self.start_time = datetime.now()

        # Create WebSocket manager (testnet=False for mainnet)
        ws_manager = LighterWebSocketManager(testnet=False)

        try:
            # Connect
            print("Connecting to Lighter WebSocket...")
            await ws_manager.connect()
            print("✓ Connected\n")

            # Subscribe to order books
            print("Subscribing to channels:")
            print("  - BTC-USDC orderbook (market_id=0)")
            await ws_manager.subscribe_orderbook(
                market_id=0, handler=lambda msg: self.capture_handler(msg, "orderbook")
            )

            print("  - ETH-USDC orderbook (market_id=1)")
            await ws_manager.subscribe_orderbook(
                market_id=1, handler=lambda msg: self.capture_handler(msg, "orderbook")
            )

            # Subscribe to trades
            print("  - BTC-USDC trades (market_id=0)")
            await ws_manager.subscribe_trades(
                market_id=0, handler=lambda msg: self.capture_handler(msg, "trades")
            )

            print("  - ETH-USDC trades (market_id=1)")
            await ws_manager.subscribe_trades(
                market_id=1, handler=lambda msg: self.capture_handler(msg, "trades")
            )

            # Subscribe to market stats
            print("  - Market stats (all)")
            await ws_manager.subscribe_market_stats(
                market_id="all", handler=lambda msg: self.capture_handler(msg, "market_stats")
            )

            print(f"\n{'='*70}")
            print(f"Capturing messages for {duration_seconds} seconds...")
            print(f"{'='*70}\n")

            # Capture for specified duration
            await asyncio.sleep(duration_seconds)

        finally:
            # Disconnect
            print(f"\n{'='*70}")
            print("Disconnecting...")
            await ws_manager.disconnect()
            print("✓ Disconnected")

        # Save samples
        self.save_samples()

    def save_samples(self):
        """Save captured samples to files"""
        print(f"\n{'='*70}")
        print("Saving captured samples...")
        print(f"{'='*70}\n")

        for category, messages in self.samples.items():
            if not messages:
                print(f"  {category:<20} - No messages captured")
                continue

            # Save all messages to one file
            output_file = self.output_dir / f"{category}_samples.json"
            with open(output_file, "w") as f:
                json.dump(messages, f, indent=2)

            print(f"  {category:<20} - {len(messages):3d} messages → {output_file.name}")

            # Save individual sample files (first few messages)
            sample_dir = self.output_dir / category
            sample_dir.mkdir(exist_ok=True)

            for i, message in enumerate(messages[:10], 1):  # Save first 10
                sample_file = sample_dir / f"sample_{i:02d}.json"
                with open(sample_file, "w") as f:
                    json.dump(message, f, indent=2)

        # Save summary
        summary = {
            "captured_at": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
            if self.start_time
            else 0,
            "total_messages": self.message_count,
            "by_category": {cat: len(msgs) for cat, msgs in self.samples.items()},
        }

        summary_file = self.output_dir / "capture_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nTotal messages captured: {self.message_count}")
        print(f"Summary saved to: {summary_file.name}")
        print(f"\n{'='*70}\n")


async def main():
    """Main entry point"""
    # Output directory
    output_dir = Path(__file__).parent.parent / "tests/qubx/connectors/xlighter/test_data/samples"

    # Create capturer
    capturer = LighterSampleCapture(output_dir)

    # Run capture (30 seconds by default, can adjust)
    await capturer.run(duration_seconds=30)


if __name__ == "__main__":
    asyncio.run(main())
