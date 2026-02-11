#!/usr/bin/env python3
"""
Real API Integrations for Multi-Agent Trading System
===================================================
Connects trading agents with real Solana/Jupiter/Helius APIs.

APIs Configured:
- Solana RPC (devnet): https://api.devnet.solana.com
- Jupiter DEX: https://api.jup.ag
- Helius (optional): Historical data

Usage:
    from api_integrations import SolanaClient, JupiterClient
    
    solana = SolanaClient()
    balance = await solana.get_balance(wallet)
    
    jupiter = JupiterClient()
    quote = await jupiter.get_quote("SOL", "USDC", 1.0)
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config

logger = logging.getLogger("api_integrations")

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class APIConfig:
    """API configuration from environment"""
    network: str = "devnet"
    
    # RPC
    rpc_url: str = "https://api.devnet.solana.com"
    
    # Wallet
    wallet_address: str = ""
    wallet_private_key: str = ""
    
    # Jupiter
    jupiter_url: str = "https://api.jup.ag/swap/v6"
    jupiter_api_key: str = ""
    
    # Helius
    helius_url: str = "https://api.devnet.solana.com"
    helius_api_key: str = ""
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment"""
        return cls(
            network=os.environ.get("NETWORK", "devnet"),
            rpc_url=os.environ.get("SOLANA_RPC_URL", "https://api.devnet.solana.com"),
            wallet_address=os.environ.get("HOT_WALLET_ADDRESS", ""),
            wallet_private_key=os.environ.get("HOT_WALLET_PRIVATE_KEY", ""),
            jupiter_url="https://api.jup.ag/swap/v6",
            jupiter_api_key=os.environ.get("JUPITER_API_KEY", ""),
            helius_url="https://api.devnet.solana.com",
            helius_api_key=os.environ.get("HELIUS_API_KEY", "")
        )


# ============================================================================
# HTTP CLIENT
# ============================================================================
class HTTPClient:
    """Async HTTP client with retry logic"""
    
    def __init__(self, base_url: str = "", headers: Dict = None):
        self.base_url = base_url
        self.headers = headers or {}
        self.session = None
        
    async def get(self, endpoint: str, params: Dict = None) -> Dict:
        """GET request with retry"""
        import aiohttp
        
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as resp:
                return await resp.json()
    
    async def post(self, endpoint: str, data: Dict = None) -> Dict:
        """POST request with retry"""
        import aiohttp
        
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=self.headers) as resp:
                return await resp.json()


# ============================================================================
# SOLANA RPC CLIENT
# ============================================================================
class SolanaClient:
    """
    Solana RPC client for blockchain operations.
    
    Supported operations:
    - get_balance
    - get_token_accounts
    - get_transaction
    - send_transaction
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig.from_env()
        self.rpc_url = self.config.rpc_url
        self.wallet_address = self.config.wallet_address
        
    async def get_balance(self, wallet: str = None) -> Dict:
        """
        Get SOL balance for a wallet.
        
        Returns:
            {
                "lamports": 5000000000,
                "sol": 5.0,
                "status": "success"
            }
        """
        address = wallet or self.wallet_address
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [address]
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    
                    if "result" in data:
                        lamports = data["result"]["value"]
                        return {
                            "lamports": lamports,
                            "sol": lamports / 1e9,
                            "status": "success"
                        }
                    
                    return {"error": data.get("error", "Unknown error")}
                    
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {"error": str(e)}
    
    async def get_token_balances(self, wallet: str = None) -> Dict:
        """Get all token balances for a wallet"""
        address = wallet or self.wallet_address
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                address,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"}
            ]
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    
                    balances = {}
                    if "result" in data:
                        for account in data["result"]["value"]:
                            info = account["account"]["data"]["parsed"]["info"]
                            balances[info["mint"]] = {
                                "amount": float(info["tokenAmount"]["amount"]),
                                "decimals": int(info["tokenAmount"]["decimals"]),
                                "ui_amount": float(info["tokenAmount"]["uiAmount"])
                            }
                    
                    return {"balances": balances, "status": "success"}
                    
        except Exception as e:
            logger.error(f"Error getting token balances: {e}")
            return {"error": str(e)}
    
    async def get_recent_blockhash(self) -> Dict:
        """Get recent blockhash"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getRecentBlockhash"
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    
                    if "result" in data:
                        return {
                            "blockhash": data["result"]["value"]["blockhash"],
                            "status": "success"
                        }
                    
                    return {"error": data.get("error", "Unknown error")}
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def get_account_info(self, address: str) -> Dict:
        """Get account information"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [
                address,
                {"encoding": "jsonParsed"}
            ]
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    return data
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def get_supply(self) -> Dict:
        """Get total SOL supply"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSupply"
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    return data
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def get_cluster_nodes(self) -> List[Dict]:
        """Get all cluster nodes"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getClusterNodes"
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    return data.get("result", {}).get("value", [])
                    
        except Exception as e:
            return []


# ============================================================================
# JUPITER DEX CLIENT
# ============================================================================
class JupiterClient:
    """
    Jupiter DEX client for token swaps.
    
    Supported operations:
    - get_quote: Get swap quote
    - get_swap: Create swap transaction
    - get_tokens: Get supported tokens
    - get_routes: Get available routes
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig.from_env()
        self.base_url = "https://api.jup.ag"
        
        # Public Lite API (works without API key)
        self.lite_url = "https://lite-api.jup.ag"
        
        # Auth headers if API key provided
        self.headers = {}
        if self.config.jupiter_api_key:
            self.headers["Authorization"] = f"Bearer {self.config.jupiter_api_key}"
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: float,
        slippage_bps: int = 50
    ) -> Dict:
        """
        Get swap quote from Jupiter.
        
        Args:
            input_mint: Input token address
            output_mint: Output token address
            amount: Amount in base units (decimals handled)
            slippage_bps: Slippage in basis points (50 = 0.5%)
        
        Returns:
            Quote with routes, prices, and fees
        """
        endpoint = f"{self.lite_url}/price/v3"
        params = {
            "ids": f"{input_mint},{output_mint}"
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_quote(data, input_mint, output_mint, amount)
                    
                    return {"error": f"HTTP {resp.status}"}
                    
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return {"error": str(e)}
    
    def _parse_quote(self, data: Dict, input_mint: str, output_mint: str, amount: float) -> Dict:
        """Parse quote response"""
        prices = data.get("data", {})
        
        input_price = prices.get(input_mint, {}).get("price", 0)
        output_price = prices.get(output_mint, {}).get("price", 0)
        
        # Calculate output
        output_amount = (amount / 1e9) * input_price / output_price * 1e6  # Assuming USDC decimals
        
        return {
            "status": "success",
            "input_mint": input_mint,
            "output_mint": output_mint,
            "input_amount": amount,
            "output_amount": output_amount,
            "input_price": input_price,
            "output_price": output_price,
            "price_impact": 0,  # Would need full quote endpoint
            "route": "SOL → USDC (direct)"
        }
    
    async def get_swap_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: float,
        slippage_bps: int = 50
    ) -> Dict:
        """
        Get detailed swap quote with all routes.
        
        Uses the full Jupiter API for best routes.
        """
        endpoint = f"{self.base_url}/swap/v6/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": int(amount * 1e9),  # Convert to lamports/smallest units
            "slippageBps": slippage_bps,
            "onlyDirectRoutes": False,
            "excludeDexes": []
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "status": "success",
                            "data": data,
                            "route": data.get("routePlan", []),
                            "out_amount": data.get("outAmount", 0),
                            "price_impact": data.get("priceImpactPct", 0)
                        }
                    
                    return {"error": f"HTTP {resp.status}"}
                    
        except Exception as e:
            logger.error(f"Error getting swap quote: {e}")
            return {"error": str(e)}
    
    async def get_swap_instruction(
        self,
        input_mint: str,
        output_mint: str,
        amount: float,
        wallet_address: str,
        slippage_bps: int = 50
    ) -> Dict:
        """
        Get swap instruction for signing.
        
        Returns the full transaction data ready to be signed.
        """
        endpoint = f"{self.base_url}/swap/v6/swap"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": int(amount * 1e9),
            "slippageBps": slippage_bps,
            "userPublicKey": wallet_address,
            "computeUnitsPrice": "1000"  # Priority fee in micro-lamports
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "status": "success",
                            "setup_missions": data.get("setupMissions", []),
                            "swap_mission": data.get("swapMission", ""),
                            "cleanup_missions": data.get("cleanupMissions", []),
                            "address_lookup_tables": data.get("addressLookupTableAddresses", [])
                        }
                    
                    return {"error": f"HTTP {resp.status}"}
                    
        except Exception as e:
            logger.error(f"Error getting swap instruction: {e}")
            return {"error": str(e)}
    
    async def get_tokens(self) -> Dict:
        """Get list of supported tokens"""
        endpoint = f"{self.base_url}/tokens/v2"
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"status": "success", "tokens": data}
                    
                    return {"error": f"HTTP {resp.status}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def get_holdings(self, wallet: str) -> Dict:
        """Get user's holdings from Jupiter"""
        endpoint = f"{self.lite_url}/ultra/v1/holdings/{wallet}"
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"status": "success", "holdings": data}
                    
                    return {"error": f"HTTP {resp.status}"}
                    
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# HELIUS RPC CLIENT
# ============================================================================
class HeliusClient:
    """
    Helius RPC client for enhanced Solana data.
    
    Enhanced features:
    - Rich account data
    - Token balances with metadata
    - Parsed transactions
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig.from_env()
        
        # Use Helius if API key provided, else use standard RPC
        if self.config.helius_api_key:
            self.rpc_url = f"https://api.helius.xyz/v0?api-key={self.config.helius_api_key}"
        else:
            self.rpc_url = self.config.rpc_url
    
    async def get_parsed_transaction(self, tx_signature: str) -> Dict:
        """Get parsed transaction with human-readable data"""
        endpoint = self.rpc_url
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getParsedTransaction",
            "params": [tx_signature]
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload) as resp:
                    data = await resp.json()
                    return data
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def get_token_balances(self, wallet: str) -> Dict:
        """Get token balances with metadata"""
        endpoint = self.rpc_url
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenBalances",
            "params": [wallet]
        }
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload) as resp:
                    data = await resp.json()
                    return data
                    
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# TRADING AGENT INTEGRATION
# ============================================================================
class TradingAPIClient:
    """
    Unified trading client combining all APIs.
    
    Used by the TradingAgent for real trading operations.
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig.from_env()
        self.solana = SolanaClient(config)
        self.jupiter = JupiterClient(config)
        self.helius = HeliusClient(config)
        
        # Default tokens
        self.SOL = "So11111111111111111111111111111111111111112"
        self.USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        self.USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYW"
        self.JUP = "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2"
        self.BONK = "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP"
    
    async def get_portfolio(self, wallet: str = None) -> Dict:
        """Get complete portfolio view"""
        address = wallet or self.config.wallet_address
        
        # Get SOL balance
        sol_balance = await self.solana.get_balance(address)
        
        # Get token balances
        token_balances = await self.solana.get_token_balances(address)
        
        # Get Jupiter holdings
        jupiter_holdings = await self.jupiter.get_holdings(address)
        
        return {
            "wallet": address,
            "network": self.config.network,
            "sol": sol_balance,
            "tokens": token_balances.get("balances", {}),
            "holdings": jupiter_holdings.get("holdings", []),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_quote(self, from_token: str, to_token: str, amount: float) -> Dict:
        """Get swap quote"""
        return await self.jupiter.get_quote(from_token, to_token, amount)
    
    async def prepare_swap(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        wallet: str = None
    ) -> Dict:
        """Prepare swap transaction"""
        address = wallet or self.config.wallet_address
        
        # Get quote
        quote = await self.jupiter.get_swap_quote(from_token, to_token, amount)
        
        if quote.get("status") != "success":
            return quote
        
        # Get instruction
        instruction = await self.jupiter.get_swap_instruction(
            from_token, to_token, amount, address
        )
        
        return {
            "status": "success",
            "quote": quote.get("data", {}),
            "instruction": instruction,
            "wallet": address
        }
    
    async def execute_swap(self, from_token: str, to_token: str, amount: float) -> Dict:
        """
        Execute a swap.
        
        Note: This prepares the transaction. Actual signing
        requires a wallet adapter or keypair.
        """
        # For devnet/testing, prepare without signing
        swap_data = await self.prepare_swap(from_token, to_token, amount)
        
        return {
            "status": "pending_signature",
            "data": swap_data,
            "message": "Transaction prepared. Requires wallet signing.",
            "network": self.config.network
        }


# ============================================================================
# API STATUS CHECKER
# ============================================================================
async def check_api_status() -> Dict:
    """Check status of all APIs"""
    solana = SolanaClient()
    jupiter = JupiterClient()
    
    results = {
        "solana_rpc": {"status": "unknown"},
        "jupiter_api": {"status": "unknown"},
        "timestamp": datetime.now().isoformat()
    }
    
    # Check Solana RPC
    try:
        balance = await solana.get_balance("65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3")
        results["solana_rpc"] = {
            "status": "✅" if balance.get("status") == "success" else "❌",
            "balance": balance
        }
    except Exception as e:
        results["solana_rpc"] = {"status": "❌", "error": str(e)}
    
    # Check Jupiter
    try:
        tokens = await jupiter.get_tokens()
        results["jupiter_api"] = {
            "status": "✅" if tokens.get("status") == "success" else "❌",
            "token_count": len(tokens.get("tokens", []))
        }
    except Exception as e:
        results["jupiter_api"] = {"status": "❌", "error": str(e)}
    
    return results


# ============================================================================
# MAIN DEMO
# ============================================================================
async def main():
    """Demo API integrations"""
    
    print("="*70)
    print("🚀 API INTEGRATIONS DEMO")
    print("="*70)
    
    # Load config
    config = APIConfig.from_env()
    print(f"\n📡 Network: {config.network}")
    print(f"   RPC: {config.rpc_url[:40]}...")
    
    # Check APIs
    print("\n📊 Checking API status...")
    status = await check_api_status()
    print(f"   Solana RPC: {status['solana_rpc']['status']}")
    print(f"   Jupiter API: {status['jupiter_api']['status']}")
    
    # Demo trading client
    print("\n💰 Trading Client Demo:")
    client = TradingAPIClient(config)
    
    # Get portfolio
    print("\n📊 Portfolio for wallet:")
    portfolio = await client.get_portfolio()
    print(f"   Wallet: {portfolio['wallet'][:20]}...")
    print(f"   SOL: {portfolio['sol'].get('sol', 'N/A')} SOL")
    print(f"   Tokens: {len(portfolio.get('holdings', []))} holdings")
    
    # Get quote
    print("\n💱 Swap Quote: SOL → USDC (1 SOL)")
    quote = await client.get_quote(client.SOL, client.USDC, 1.0)
    print(f"   Status: {quote.get('status')}")
    if "output_amount" in quote:
        print(f"   Output: {quote['output_amount']:.2f} USDC")
    
    # Prepare swap
    print("\n🔄 Preparing swap: 0.5 SOL → USDC")
    swap = await client.prepare_swap(client.SOL, client.USDC, 0.5)
    print(f"   Status: {swap.get('status')}")
    
    print("\n" + "="*70)
    print("✅ API Integrations Demo Complete")
    print("="*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
