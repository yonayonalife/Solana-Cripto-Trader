#!/usr/bin/env python3
"""
Jupiter API V6 Client for Solana Trading Bot
"""
import asyncio
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass
from decimal import Decimal

# Constants
JUPITER_BASE_URL = "https://api.jup.ag/swap/v6"

# Token Mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW"
JUP_MINT = "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP"


@dataclass
class Quote:
    """Jupiter Quote Response"""
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    other_amount_threshold: int
    swap_mode: str
    slippage_bps: int
    price_impact_pct: float
    platform_fee_amount: int
    route_percent: int
    
    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "Quote":
        platform_fee = data.get("platformFee", {})
        return cls(
            input_mint=data["inputMint"],
            output_mint=data["outputMint"],
            in_amount=int(data["inAmount"]),
            out_amount=int(data["outAmount"]),
            other_amount_threshold=int(data["otherAmountThreshold"]),
            swap_mode=data.get("swapMode", "ExactIn"),
            slippage_bps=data.get("slippageBps", 0),
            price_impact_pct=float(data.get("priceImpactPct", "0")),
            platform_fee_amount=int(platform_fee.get("amount", 0)),
            route_percent=100  # Simplified
        )
    
    def get_output_with_slippage(self) -> int:
        """Get minimum output considering slippage"""
        return self.other_amount_threshold


@dataclass
class SwapData:
    """Jupiter Swap Response"""
    swap_transaction: str
    last_valid_block_height: int
    
    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> "SwapData":
        return cls(
            swap_transaction=data["swapTransaction"],
            last_valid_block_height=data.get("lastValidBlockHeight", 0)
        )


class JupiterClient:
    """Client for Jupiter V6 API"""
    
    def __init__(self, base_url: str = JUPITER_BASE_URL):
        self.base_url = base_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        swap_mode: str = "ExactIn"
    ) -> Quote:
        """
        Get quote for swap
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in lamports
            slippage_bps: Slippage in basis points (default: 50 = 0.5%)
            swap_mode: 'ExactIn' or 'ExactOut'
        
        Returns:
            Quote object with swap details
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "swapMode": swap_mode
        }
        
        response = await self.http_client.get(
            f"{self.base_url}/quote",
            params=params
        )
        response.raise_for_status()
        
        return Quote.from_response(response.json())
    
    async def create_swap(
        self,
        quote: Quote,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
        priority_fee_lamports: Optional[int] = None,
        jito_tip_lamports: int = 0
    ) -> SwapData:
        """
        Create swap transaction
        
        Args:
            quote: Quote from get_quote
            user_public_key: Wallet public key
            wrap_unwrap_sol: Auto wrap/unwrap SOL
            priority_fee_lamports: Priority fee in lamports
            jito_tip_lamports: Jito tip in lamports
        
        Returns:
            SwapData with serialized transaction
        """
        # Build prioritization fee config
        prioritization_fee = {
            "global": True
        }
        
        if priority_fee_lamports:
            prioritization_fee["priorityLevelWithMaxLamports"] = {
                "medium": priority_fee_lamports
            }
        
        payload = {
            "quoteResponse": {
                "inputMint": quote.input_mint,
                "inAmount": str(quote.in_amount),
                "outputMint": quote.output_mint,
                "outAmount": str(quote.out_amount),
                "otherAmountThreshold": str(quote.other_amount_threshold),
                "swapMode": quote.swap_mode,
                "slippageBps": str(quote.slippage_bps),
                "platformFee": {
                    "amount": str(quote.platform_fee_amount),
                    "feeBps": "0",
                    "feeMint": quote.output_mint
                }
            },
            "userPublicKey": user_public_key,
            "wrapUnwrapSOL": wrap_unwrap_sol,
            "prioritizationFeeLamports": prioritization_fee
        }
        
        # Add Jito tip if specified
        if jito_tip_lamports > 0:
            payload["jitoTipLamports"] = str(jito_tip_lamports)
        
        response = await self.http_client.post(
            f"{self.base_url}/swap",
            json=payload
        )
        response.raise_for_status()
        
        return SwapData.from_response(response.json())
    
    async def get_price(
        self,
        ids: list[str],
        vs_token: str = "USDC"
    ) -> Dict[str, float]:
        """
        Get price for tokens
        
        Args:
            ids: List of token mint addresses
            vs_token: Quote currency (default: USDC)
        
        Returns:
            Dict mapping mint -> price
        """
        params = {
            "ids": ",".join(ids),
            "vsToken": vs_token
        }
        
        response = await self.http_client.get(
            "https://api.jup.ag/price/v2",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        return {k: float(v["value"]) for k, v in data.get("data", {}).items()}
    
    async def get_tokens(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of supported tokens
        
        Returns:
            Dict mapping mint -> token info
        """
        response = await self.http_client.get(
            "https://api.jup.ag/tokens"
        )
        response.raise_for_status()
        
        return {t["address"]: t for t in response.json()}


# Utility functions
def decimal_to_lamports(amount: float, decimals: int) -> int:
    """Convert decimal amount to lamports"""
    return int(amount * (10 ** decimals))


def lamports_to_decimal(lamports: int, decimals: int) -> float:
    """Convert lamports to decimal amount"""
    return lamports / (10 ** decimals)


def calculate_fees(
    quote: Quote,
    priority_fee_lamports: int = 1000,
    jito_tip_lamports: int = 0,
    network_fee_lamports: int = 5000
) -> Dict[str, float]:
    """
    Calculate total fees for a swap
    
    Args:
        quote: Jupiter quote
        priority_fee_lamports: Priority fee
        jito_tip_lamports: Jito tip
        network_fee_lamports: Base network fee
    
    Returns:
        Dict with fee breakdown in SOL and USD
    """
    # Jupiter route fee
    route_fee_lamports = quote.platform_fee_amount
    
    # Priority fee
    priority_fee = priority_fee_lamports
    
    # Jito tip
    jito_fee = jito_tip_lamports
    
    # Network fee (estimated)
    network_fee = network_fee_lamports
    
    # Total in lamports
    total_lamports = network_fee + route_fee_lamports + priority_fee + jito_fee
    
    # Convert to SOL (1 SOL = 1e9 lamports)
    total_sol = total_lamports / 1_000_000_000
    
    # Estimate USD value (assuming SOL ~$100)
    sol_price = 100.0  # Should fetch real price
    total_usd = total_sol * sol_price
    
    return {
        "network_fee_lamports": network_fee,
        "route_fee_lamports": route_fee_lamports,
        "priority_fee_lamports": priority_fee,
        "jito_fee_lamports": jito_fee,
        "total_fee_lamports": total_lamports,
        "total_fee_sol": total_sol,
        "total_fee_usd": total_usd,
        "fee_percentage": (total_lamports / quote.in_amount) * 100
    }


# Example usage
async def main():
    async with JupiterClient() as client:
        # Get quote for 1 SOL -> USDC
        quote = await client.get_quote(
            input_mint=SOL_MINT,
            output_mint=USDC_MINT,
            amount=decimal_to_lamports(1.0, 9),
            slippage_bps=50
        )
        
        print(f"Input: {lamports_to_decimal(quote.in_amount, 9)} SOL")
        print(f"Output: {lamports_to_decimal(quote.out_amount, 6)} USDC")
        print(f"Price Impact: {quote.price_impact_pct}%")
        
        # Calculate fees
        fees = calculate_fees(quote, priority_fee_lamports=1000)
        print(f"\nFees Breakdown:")
        print(f"  Network: {fees['network_fee_lamports']} lamports")
        print(f"  Jupiter Route: {fees['route_fee_lamports']} lamports")
        print(f"  Priority: {fees['priority_fee_lamports']} lamports")
        print(f"  Total: {fees['total_fee_sol']:.6f} SOL ({fees['total_fee_usd']:.2f} USD)")


if __name__ == "__main__":
    asyncio.run(main())
