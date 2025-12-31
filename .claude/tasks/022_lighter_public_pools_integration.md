# 022: Lighter Public Pools Integration

## Overview

This proposal outlines how to create, manage, and interact with **Public Pools** on the Lighter exchange. Public pools allow you to act as an **operator** who manages a trading fund that external users can deposit into and share in the trading profits/losses.

## What is a Lighter Public Pool?

A Public Pool on Lighter is essentially a **managed trading account** where:
- **Operator** (you): Creates the pool, trades on behalf of depositors, earns operator fees
- **Depositors** (others): Deposit USDC to receive shares, benefit from operator's trading performance
- **Shares**: Represent proportional ownership of the pool's total asset value

### Key Concepts

| Concept | Description |
|---------|-------------|
| `pool_account_index` | Unique identifier for the pool (separate from your user account) |
| `operator_fee` | Performance fee charged by operator (in basis points × 1000, e.g., 100000 = 10%) |
| `initial_total_shares` | Initial shares when creating pool (1000 shares ≈ 1 USDC, so 1_000_000 = 1000 USDC min deposit) |
| `min_operator_share_rate` | Minimum % of shares operator must hold (e.g., 100 = 1%, 1000 = 10%) |
| `status` | Pool state: 0 = active, 1 = frozen |
| `share_price` | `total_asset_value / total_shares` - fluctuates with trading performance |

## Lighter SDK Capabilities

The Lighter Python SDK (`/home/yuriy/devs/lighter-python`) provides full support for public pools via `SignerClient`:

### Pool Management Methods

```python
# Create a new public pool
await client.create_public_pool(
    operator_fee=100000,           # 10% operator fee
    initial_total_shares=1_000_000, # 1000 USDC initial deposit
    min_operator_share_rate=100,    # 1% minimum operator share
)

# Update pool settings (operator_fee can ONLY decrease)
await client.update_public_pool(
    public_pool_index=pool_account_index,
    status=0,                       # 0=active, 1=frozen
    operator_fee=50000,             # 5% (reduced from 10%)
    min_operator_share_rate=1000,   # 10%
)
```

### Deposit/Withdraw Methods

```python
# Deposit (mint shares) - transfers USDC from user account to pool
await client.mint_shares(
    public_pool_index=pool_account_index,
    share_amount=10_000,  # Amount of shares to mint
)

# Withdraw (burn shares) - transfers USDC from pool back to user
await client.burn_shares(
    public_pool_index=pool_account_index,
    share_amount=10_000,  # Amount of shares to burn
)
```

### Query Methods

```python
# Get all public pools metadata
account_api = lighter.AccountApi(api_client)
pools = await account_api.public_pools_metadata()
for pool in pools.public_pools:
    print(f"Pool {pool.account_index}: APY={pool.annual_percentage_yield}%, Sharpe={pool.sharpe_ratio}")

# Get pool details (pools are accounts with pool_info)
pool_resp = await account_api.account(by="index", value=str(pool_account_index))
pool_account = pool_resp.accounts[0]
share_price = float(pool_account.total_asset_value) / float(pool_account.pool_info.total_shares)

# Get user's shares in pools
account = await account_api.account(by="index", value=str(user_account_index))
for share in account.accounts[0].shares:
    print(f"Pool {share.public_pool_index}: {share.shares_amount} shares, entry={share.entry_usdc}")
```

## Current Qubx xlighter Connector Status

The existing Qubx xlighter connector (`src/qubx/connectors/xlighter/`) handles:
- Account WebSocket subscriptions (positions, orders, balances)
- Order management (create, cancel, modify)
- Position tracking
- Balance/margin tracking

**Missing**: No public pool functionality is currently implemented.

## Implementation Options

### Option A: Standalone Pool Manager (Recommended for Initial Use)

Create a simple utility class/script outside of Qubx core for pool management. This is ideal for your immediate needs.

**Pros**: Quick to implement, doesn't require Qubx core changes
**Cons**: Not integrated with Qubx strategy framework

```python
# Example: lighter_pool_manager.py
import asyncio
import lighter
from lighter.signer_client import SignerClient

class LighterPoolManager:
    def __init__(self, url: str, account_index: int, api_keys: dict):
        self.client = SignerClient(url, account_index, api_keys)
        self.api_client = lighter.ApiClient(configuration=lighter.Configuration(host=url))
        self.account_api = lighter.AccountApi(self.api_client)

    async def create_pool(self, operator_fee_pct: float, initial_usdc: float, min_operator_pct: float):
        """Create a new public pool."""
        err = self.client.check_client()
        if err:
            raise Exception(f"Client check failed: {err}")

        tx_info, response, err = await self.client.create_public_pool(
            operator_fee=int(operator_fee_pct * 10000),  # Convert % to basis points * 1000
            initial_total_shares=int(initial_usdc * 1000),  # Convert USDC to shares
            min_operator_share_rate=int(min_operator_pct * 100),  # Convert % to basis points
        )
        if err:
            raise Exception(f"Failed to create pool: {err}")
        return response.tx_hash

    async def list_pools(self):
        """List all public pools."""
        pools = await self.account_api.public_pools_metadata()
        return pools.public_pools

    async def get_pool_info(self, pool_index: int):
        """Get detailed pool info."""
        resp = await self.account_api.account(by="index", value=str(pool_index))
        return resp.accounts[0]

    async def deposit(self, pool_index: int, usdc_amount: float):
        """Deposit USDC to a pool (mint shares)."""
        # Get current share price
        pool = await self.get_pool_info(pool_index)
        share_price = float(pool.total_asset_value) / float(pool.pool_info.total_shares)
        shares_to_mint = int(usdc_amount / share_price)

        tx_info, response, err = await self.client.mint_shares(
            public_pool_index=pool_index,
            share_amount=shares_to_mint,
        )
        if err:
            raise Exception(f"Failed to mint shares: {err}")
        return response.tx_hash

    async def withdraw(self, pool_index: int, shares_amount: int):
        """Withdraw from a pool (burn shares)."""
        tx_info, response, err = await self.client.burn_shares(
            public_pool_index=pool_index,
            share_amount=shares_amount,
        )
        if err:
            raise Exception(f"Failed to burn shares: {err}")
        return response.tx_hash

    async def get_my_shares(self):
        """Get all pool shares owned by this account."""
        resp = await self.account_api.account(by="index", value=str(self.client.account_index))
        return resp.accounts[0].shares

    async def close(self):
        await self.client.close()
        await self.api_client.close()
```

### Option B: Qubx Integration (Future Enhancement)

Integrate pool management into the Qubx xlighter connector. This would allow:
- Running strategies on pool accounts
- Automatic pool state tracking
- Integration with Qubx's position/order management

**Implementation steps:**
1. Add pool-related methods to `LighterClient`
2. Support pool accounts in `LighterAccountProcessor`
3. Add pool-specific configuration in strategy YAML

### Option C: Separate Pool Strategy Account

Use Qubx to run strategies on the pool account itself:
1. Create pool manually using Option A
2. Configure Qubx with pool's `account_index` instead of user's
3. Strategy trades execute on pool's behalf

**Note**: This requires that the pool account has API keys registered, which may need special handling.

## Recommended Workflow

### Step 1: Create Your Pool

```python
import asyncio
from lighter_pool_manager import LighterPoolManager

async def main():
    manager = LighterPoolManager(
        url="https://mainnet.zklighter.elliot.ai",
        account_index=YOUR_ACCOUNT_INDEX,
        api_keys={0: "YOUR_API_PRIVATE_KEY"}
    )

    # Create pool with 10% operator fee, 1000 USDC initial, 5% min operator share
    tx_hash = await manager.create_pool(
        operator_fee_pct=10.0,
        initial_usdc=1000.0,
        min_operator_pct=5.0
    )
    print(f"Pool created! TX: {tx_hash}")

    # Wait for tx to confirm and get pool account index
    # (pool_account_index is returned in tx event_info['a'])

    await manager.close()

asyncio.run(main())
```

### Step 2: Fund Your Pool

Transfer USDC from your main account to the pool, or deposit via `mint_shares`.

### Step 3: Trade on Pool Account

Either:
- Use Qubx configured with pool's account_index
- Trade manually via Lighter SDK
- Use custom trading bot

### Step 4: Monitor Performance

```python
async def monitor_pool(manager, pool_index):
    pool = await manager.get_pool_info(pool_index)
    print(f"Total Asset Value: {pool.total_asset_value}")
    print(f"Total Shares: {pool.pool_info.total_shares}")
    print(f"APY: {pool.pool_info.annual_percentage_yield}%")
    print(f"Sharpe Ratio: {pool.pool_info.sharpe_ratio}")
    print(f"Operator Fee: {pool.pool_info.operator_fee}")
    print(f"Operator Shares: {pool.pool_info.operator_shares}")
```

## Data Models Reference

### PublicPoolMetadata (from API)
```python
{
    "account_index": int,           # Pool's unique identifier
    "created_at": int,              # Creation timestamp
    "master_account_index": int,    # Operator's main account
    "account_type": int,            # Account type (pool = special value)
    "name": str,                    # Pool name
    "l1_address": str,              # L1 address
    "annual_percentage_yield": float,  # APY %
    "sharpe_ratio": float,          # Sharpe ratio
    "status": int,                  # 0=active, 1=frozen
    "operator_fee": str,            # Operator fee (basis points string)
    "total_asset_value": str,       # Total USDC value
    "total_shares": int,            # Total shares outstanding
}
```

### PublicPoolInfo (from account details)
```python
{
    "status": int,                  # 0=active, 1=frozen
    "operator_fee": str,            # Operator fee
    "min_operator_share_rate": str, # Min operator share %
    "total_shares": int,            # Total shares
    "operator_shares": int,         # Operator's shares
    "annual_percentage_yield": float,
    "sharpe_ratio": float,
    "daily_returns": [...],         # Historical daily returns
    "share_prices": [...],          # Historical share prices
}
```

### PublicPoolShare (user's shares)
```python
{
    "public_pool_index": int,  # Pool identifier
    "shares_amount": int,      # Number of shares owned
    "entry_usdc": str,         # USDC amount at entry
}
```

## Important Notes

1. **Operator Fee Can Only Decrease**: Once set, you can only lower the operator fee, never increase it.

2. **Min Operator Share Rate**: You must maintain at least this % of total shares. If you try to withdraw below this, it will fail.

3. **Share Pricing**: Share price = total_asset_value / total_shares. Price changes with trading P&L.

4. **Initial Deposit**: When creating a pool, the `initial_total_shares` amount is deducted from your account.

5. **Pool Trading**: The pool account can trade like any other account. Positions, orders, and P&L affect the pool's total_asset_value.

6. **Depositor Entry/Exit**: When users mint shares, they pay current share price. When they burn shares, they receive current share price.

## Questions to Clarify

1. Do you want to create a pool and run Qubx strategies on it, or manage the pool separately?

2. What operator fee % do you want to charge?

3. What's your initial capital for the pool?

4. Do you need automatic rebalancing or deposit/withdrawal handling?

5. Should Qubx track pool performance metrics (APY, Sharpe, etc.)?

## Next Steps

1. **Immediate**: Create standalone `LighterPoolManager` utility for pool creation and management
2. **Short-term**: Test pool creation on testnet (if available) or mainnet with small amount
3. **Medium-term**: Integrate pool account support into Qubx xlighter connector
4. **Long-term**: Add pool-specific strategy features (auto-rebalancing, fee collection, etc.)
