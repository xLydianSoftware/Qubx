"""
Debug script to capture live Lighter account WebSocket messages.

This script connects to Lighter WebSocket with your account credentials,
subscribes to account-related channels, and captures all messages in real-time.

Use this to debug why account processor isn't receiving updates and to capture
actual message formats from Lighter exchange.

Usage:
    # Run until Ctrl+C (recommended for manual trading)
    poetry run python scripts/debug_lighter_account_updates.py

    # Run for specific duration (5 minutes)
    poetry run python scripts/debug_lighter_account_updates.py --duration 300

    # Test without auth tokens (to verify if auth is required)
    poetry run python scripts/debug_lighter_account_updates.py --without-auth

Channels Captured:
    - account_tx/{account_id} - Account-specific transactions (requires auth)
    - account_all_orders/{account_id} - All orders across markets (requires auth)
    - account_all/{account_id} - Positions, trades history, funding (requires auth)
    - user_stats/{account_id} - Account statistics (requires auth)

Auth Token Info:
    - Default expiry: 10 minutes (600 seconds)
    - Token is checked during subscription handshake
    - Most likely: subscriptions stay active after initial auth
    - If running >10 minutes, watch for disconnections (may need to resubscribe)

Output:
    Saves to: tests/qubx/connectors/xlighter/test_data/account_samples/
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

from qubx import logger
from qubx.connectors.xlighter import LighterClient, LighterWebSocketManager
from qubx.utils.runner.accounts import AccountConfigurationManager


class LighterAccountCapture:
    """Capture Lighter account WebSocket messages"""

    def __init__(
        self,
        account_index: int,
        client: LighterClient,
        output_dir: Path,
        use_auth: bool = True,
    ):
        self.account_index = account_index
        self.client = client
        self.output_dir = output_dir
        self.use_auth = use_auth
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for captured samples
        self.samples = {
            "account_tx": [],
            "account_all_orders": [],
            "account_all": [],
            "user_stats": [],
            "system": [],
        }

        # Statistics
        self.message_count = 0
        self.start_time = None
        self.auth_token = None
        self.ws_manager = None
        self.should_stop = False

    async def generate_auth_token(self) -> str | None:
        """Generate authentication token using SignerClient"""
        if not self.use_auth:
            logger.info("Auth tokens disabled (--without-auth)")
            return None

        try:
            # SignerClient is already initialized in LighterClient
            signer = self.client.signer_client
            auth_token, error = signer.create_auth_token_with_expiry()

            if error:
                logger.error(f"Failed to generate auth token: {error}")
                return None

            # Calculate expiry time
            from datetime import datetime, timedelta
            expiry_time = datetime.now() + timedelta(minutes=10)
            print(f"✓ Generated auth token (expires at {expiry_time.strftime('%H:%M:%S')})")
            print(f"  Note: If running >10 min, watch for subscription drops")
            return auth_token

        except Exception as e:
            logger.error(f"Error generating auth token: {e}")
            return None

    async def capture_handler(self, message: dict, channel_type: str):
        """Handler for captured messages"""
        self.message_count += 1

        # Add metadata
        message_with_meta = {
            "captured_at": datetime.now().isoformat(),
            "message_number": self.message_count,
            "channel_type": channel_type,
            "data": message,
        }

        # Store in appropriate category
        if channel_type in self.samples:
            self.samples[channel_type].append(message_with_meta)
        else:
            self.samples["system"].append(message_with_meta)

        # Console output
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg_type = message.get("type", "unknown")
        channel = message.get("channel", "unknown")

        # Pretty print first 500 chars of message
        msg_preview = json.dumps(message, indent=2)[:500]
        if len(json.dumps(message)) > 500:
            msg_preview += "\n... (truncated)"

        print(f"\n[{self.message_count:03d}] {timestamp} | {channel} | {msg_type}")
        print(f"      {msg_preview}")

    async def subscribe_with_optional_auth(self, channel: str, handler, message_type: str):
        """
        Subscribe to a channel with optional authentication.

        Uses the WebSocket manager's subscribe method with auth parameter.
        """
        # Prepare subscription parameters
        params = {}
        if self.auth_token:
            params["auth"] = self.auth_token
            logger.info(f"  ✓ {channel} (with auth)")
        else:
            logger.info(f"  ✓ {channel} (without auth)")

        # Subscribe via WebSocket manager (which handles channel format conversion)
        await self.ws_manager.subscribe(channel, handler, **params)

    async def run(self, duration_seconds: int | None = None):
        """
        Run capture for specified duration or until interrupted.

        Args:
            duration_seconds: How long to capture (None = until Ctrl+C)
        """
        print(f"\n{'=' * 70}")
        print(f"Lighter Account WebSocket Debug Capture")
        print(f"{'=' * 70}")
        print(f"Account: {self.account_index}")
        print(f"Duration: {duration_seconds if duration_seconds else 'Until Ctrl+C'} seconds")
        print(f"Auth tokens: {'ENABLED' if self.use_auth else 'DISABLED'}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 70}\n")

        self.start_time = datetime.now()

        # Generate auth token if needed
        self.auth_token = await self.generate_auth_token()

        # Create WebSocket manager
        self.ws_manager = LighterWebSocketManager(testnet=False)

        try:
            # Connect
            print("Connecting to Lighter WebSocket...")
            await self.ws_manager.connect()
            print("✓ Connected\n")

            # Subscribe to channels
            print("Subscriptions:")

            # Subscribe to account_tx (transactions specific to this account)
            await self.subscribe_with_optional_auth(
                channel=f"account_tx/{self.account_index}",
                handler=lambda msg: self.capture_handler(msg, "account_tx"),
                message_type="account_tx",
            )

            # Subscribe to account_all_orders (all orders across markets)
            await self.subscribe_with_optional_auth(
                channel=f"account_all_orders/{self.account_index}",
                handler=lambda msg: self.capture_handler(msg, "account_all_orders"),
                message_type="account_all_orders",
            )

            # Subscribe to account_all (positions, trades history, funding)
            await self.subscribe_with_optional_auth(
                channel=f"account_all/{self.account_index}",
                handler=lambda msg: self.capture_handler(msg, "account_all"),
                message_type="account_all",
            )

            # Subscribe to user_stats (account statistics)
            await self.subscribe_with_optional_auth(
                channel=f"user_stats/{self.account_index}",
                handler=lambda msg: self.capture_handler(msg, "user_stats"),
                message_type="user_stats",
            )

            print(f"\n{'=' * 70}")
            print(f"Capturing messages... Press Ctrl+C to stop")
            print(f"{'=' * 70}")

            # Capture for specified duration or until stopped
            if duration_seconds:
                await asyncio.sleep(duration_seconds)
            else:
                # Wait indefinitely until interrupted
                while not self.should_stop:
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Capture cancelled")
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")
        finally:
            # Disconnect
            print(f"\n{'=' * 70}")
            print("Disconnecting...")
            await self.ws_manager.disconnect()
            print("✓ Disconnected")

        # Save samples
        self.save_samples()

    def save_samples(self):
        """Save captured samples to files"""
        print(f"\n{'=' * 70}")
        print("Saving captured samples...")
        print(f"{'=' * 70}\n")

        for category, messages in self.samples.items():
            if not messages:
                print(f"  {category:<25} - No messages captured")
                continue

            # Save all messages to one file
            output_file = self.output_dir / f"{category}_samples.json"
            with open(output_file, "w") as f:
                json.dump(messages, f, indent=2)

            print(f"  {category:<25} - {len(messages):3d} messages → {output_file.name}")

            # Save individual sample files (first 20 messages)
            sample_dir = self.output_dir / category
            sample_dir.mkdir(exist_ok=True)

            for i, message in enumerate(messages[:20], 1):  # Save first 20
                sample_file = sample_dir / f"sample_{i:02d}.json"
                with open(sample_file, "w") as f:
                    json.dump(message, f, indent=2)

        # Save summary
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        summary = {
            "captured_at": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": duration,
            "account_index": self.account_index,
            "auth_enabled": self.use_auth,
            "total_messages": self.message_count,
            "by_category": {cat: len(msgs) for cat, msgs in self.samples.items()},
        }

        summary_file = self.output_dir / "capture_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\nCapture Summary:")
        print(f"  Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
        print(f"  Total messages: {self.message_count}")
        for cat, count in summary["by_category"].items():
            if count > 0:
                print(f"    - {cat}: {count}")
        print(f"\nSummary saved to: {summary_file.name}")
        print(f"\n{'=' * 70}\n")

    def stop(self):
        """Signal to stop capture"""
        self.should_stop = True


async def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Capture live Lighter account WebSocket messages for debugging")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Capture duration in seconds (default: until Ctrl+C)",
    )
    parser.add_argument(
        "--without-auth",
        action="store_true",
        help="Test subscriptions without auth tokens",
    )
    parser.add_argument(
        "--account-config",
        type=str,
        default="/home/yuriy/accounts/xlydian1_lighter.toml",
        help="Path to account config file",
    )
    args = parser.parse_args()

    # Output directory
    output_dir = Path(__file__).parent.parent / "tests/qubx/connectors/xlighter/test_data/account_samples"

    # Load account configuration
    print(f"\nLoading account configuration...")
    account_config_path = Path(args.account_config)
    if not account_config_path.exists():
        print(f"Error: Account config not found: {account_config_path}")
        sys.exit(1)

    account_manager = AccountConfigurationManager(account_config=account_config_path)

    try:
        credentials = account_manager.get_exchange_credentials("LIGHTER")
        print(f"✓ Loaded credentials from {account_config_path}")
    except KeyError:
        print(f"Error: No LIGHTER credentials found in {account_config_path}")
        sys.exit(1)

    # Extract Lighter-specific fields
    account_index = credentials.get_extra_field("account_index")
    api_key_index = credentials.get_extra_field("api_key_index")

    if account_index is None:
        print("Error: 'account_index' not found in account config")
        sys.exit(1)

    print(f"  Account index: {account_index}")
    print(f"  API key index: {api_key_index}")

    # Initialize LighterClient (which includes SignerClient for auth)
    print(f"\nInitializing Lighter client...")
    try:
        client = LighterClient(
            api_key=credentials.api_key,
            private_key=credentials.secret,
            account_index=account_index,
            api_key_index=api_key_index or 1,
            testnet=False,
        )
        print(f"✓ Client initialized")
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)

    # Create capturer
    capturer = LighterAccountCapture(
        account_index=account_index,
        client=client,
        output_dir=output_dir,
        use_auth=not args.without_auth,
    )

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping capture...")
        capturer.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Run capture
    await capturer.run(duration_seconds=args.duration)

    # Cleanup
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
