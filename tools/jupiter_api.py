#!/usr/bin/env python3
"""
Jupiter API Client - Python Wrapper
Based on official Jupiter V3 Documentation
"""
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# API Endpoints
TOKENS_API = "https://lite-api.jup.ag/tokens/v2"
PRICE_API = "https://lite-api.jup.ag/price/v3"
ULTRA_API = "https://lite-api.jup.ag/ultra/v1"

# Common Token Addresses
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYW"
JUP = "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2"
BONK = "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP"


@dataclass
class TokenInfo:
    """Token metadata from Tokens V2 API"""
    id: str
    name: str
    symbol: str
    decimals: int
    icon: str
    usd_price: float
    liquidity: float
    tags: List[str]


@dataclass
class Quote:
    """Swap quote from Ultra API"""
    in_amount: int
    out_amount: int
    other_amount_threshold: int
    price_impact_pct: float
    route_plan: List[Dict]
    platform_fee: int
    swap_mode: str
    total_time_ms: int
    
    @classmethod
    def from_response(cls, data: Dict) -> "Quote":
        return cls(
            in_amount=int(data.get("inAmount", 0)),
            out_amount=int(data.get("outAmount", 0)),
            other_amount_threshold=int(data.get("otherAmountThreshold", 0)),
            price_impact_pct=float(data.get("priceImpact", 0)),
            route_plan=data.get("routePlan", []),
            platform_fee=data.get("platformFee", {}).get("amount", 0),
            swap_mode=data.get("swapMode", "ExactIn"),
            total_time_ms=data.get("totalTime", 0)
        )


class JupiterClient:
    """Python client for Jupiter APIs"""
    
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.http.aclose()
    
    # ============ TOKENS V2 API ============
    
    async def search_tokens(self, query: str) -> List[TokenInfo]:
        """Search tokens by name, symbol, or mint address"""
        url = f"{TOKENS_API}/search?query={query}"
        resp = await self.http.get(url)
        data = resp.json()
        
        tokens = []
        if isinstance(data, list):
            for item in data:
                tokens.append(TokenInfo(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    symbol=item.get("symbol", ""),
                    decimals=item.get("decimals", 0),
                    icon=item.get("icon", ""),
                    usd_price=item.get("usdPrice", 0),
                    liquidity=item.get("liquidity", 0),
                    tags=item.get("tags", [])
                ))
        return tokens
    
    async def get_verified_tokens(self) -> List[TokenInfo]:
        """Get all verified tokens"""
        url = f"{TOKENS_API}/tag?query=verified"
        resp = await self.http.get(url)
        return self._parse_tokens(resp.json())
    
    async def get_trending_tokens(self, interval: str = "1h") -> List[TokenInfo]:
        """Get trending tokens"""
        url = f"{TOKENS_API}/toptrending/{interval}"
        resp = await self.http.get(url)
        return self._parse_tokens(resp.json())
    
    def _parse_tokens(self, data: Any) -> List[TokenInfo]:
        tokens = []
        if isinstance(data, list):
            for item in data:
                tokens.append(TokenInfo(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    symbol=item.get("symbol", ""),
                    decimals=item.get("decimals", 0),
                    icon=item.get("icon", ""),
                    usd_price=item.get("usdPrice", 0),
                    liquidity=item.get("liquidity", 0),
                    tags=item.get("tags", [])
                ))
        return tokens
    
    # ============ PRICE V3 API ============
    
    async def get_prices(self, mints: List[str]) -> Dict[str, Dict]:
        """Get USD prices for tokens"""
        ids = ",".join(mints)
        url = f"{PRICE_API}?ids={ids}"
        resp = await self.http.get(url)
        return resp.json()
    
    # ============ ULTRA API ============
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        taker: Optional[str] = None
    ) -> Quote:
        """
        Get swap quote (no wallet needed for quote)
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in lamports (1 SOL = 1_000_000_000 lamports)
            taker: Optional wallet address for signed transaction
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount)
        }
        if taker:
            params["taker"] = taker
        
        url = f"{ULTRA_API}/order"
        resp = await self.http.get(url, params=params)
        data = resp.json()
        return Quote.from_response(data)
    
    async def execute_swap(
        self,
        signed_transaction: str,
        request_id: str
    ) -> Dict:
        """
        Execute a signed swap transaction
        
        Args:
            signed_transaction: Base64 encoded signed transaction
            request_id: Request ID from quote response
        """
        url = f"{ULTRA_API}/execute"
        payload = {
            "signedTransaction": signed_transaction,
            "requestId": request_id
        }
        resp = await self.http.post(url, json=payload)
        return resp.json()
    
    async def get_holdings(self, wallet_address: str) -> Dict:
        """Get wallet token balances"""
        url = f"{ULTRA_API}/holdings/{wallet_address}"
        resp = await self.http.get(url)
        return resp.json()
    
    async def get_token_warnings(self, mints: List[str]) -> Dict:
        """Get security warnings for tokens"""
        mint_list = ",".join(mints)
        url = f"{ULTRA_API}/shield?mints={mint_list}"
        resp = await self.http.get(url)
        return resp.json()


# ============ DEMO ============

async def demo():
    client = JupiterClient()
    
    print("=" * 60)
    print("🏆 JUPITER API V3 - DEMO COMPLETO")
    print("=" * 60)
    print()
    
    # 1. Search tokens
    print("1️⃣  Buscando tokens 'JUP'...")
    tokens = await client.search_tokens("JUP")
    for t in tokens[:3]:
        print(f"   {t.symbol}: ${t.usd_price}")
    print()
    
    # 2. Get prices
    print("2️⃣  Obteniendo precios...")
    prices = await client.get_prices([SOL, USDC, JUP])
    print("   Preços:")
    for addr, info in prices.items():
        symbol = addr[:8]
        price = info.get("usdPrice", 0)
        print(f"   {symbol}: ${price}")
    print()
    
    # 3. Get quote
    print("3️⃣  Quote: 1 SOL → USDC")
    quote = await client.get_quote(SOL, USDC, 1_000_000_000)  # 1 SOL
    print(f"   Input:  1.000000000 SOL")
    print(f"   Output: {quote.out_amount / 1_000_000:.2f} USDC")
    print(f"   Impact: {quote.price_impact_pct:.4f}%")
    print(f"   Time:   {quote.total_time_ms}ms")
    print()
    
    # 4. Get holdings
    print("4️⃣  Holdings de wallet...")
    holdings = await client.get_holdings("65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3")
    print(f"   Response: {str(holdings)[:200]}")
    
    await client.close()
    print()
    print("=" * 60)
    print("✅ DEMO COMPLETADO!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
