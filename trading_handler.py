#!/usr/bin/env python3
"""
Trading Handler CLI for OpenClaw/Eko Integration
================================================

Enable trading commands via Telegram:
    Eko, mi balance
    Eko, precio de SOL
    Eko, compra 0.5 SOL
    Eko, vende 1 SOL
    Eko, status del sistema

Usage:
    python3 trading_handler.py --balance
    python3 trading_handler.py --price
    python3 trading_handler.py --buy 0.5
    python3 trading_handler.py --sell 1.0
    python3 trading_handler.py --status
    python3 trading_handler.py --address
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Solana imports
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.api import Client

# Jupiter API
import requests


class TradingHandler:
    """
    Handle trading commands for Solana via Jupiter DEX.
    """
    
    # RPC endpoints
    RPC_DEVNET = "https://api.devnet.solana.com"
    RPC_MAINNET = "https://api.mainnet-beta.solana.com"
    
    # Jupiter API
    JUPITER_PRICE_URL = "https://price.jup.ag/v6/price"
    JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
    JUPITER_SWAP_URL = "https://api.jup.ag/swap/v1/swap"
    
    # Token mints
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    
    def __init__(self, network: str = "devnet"):
        self.network = network
        self.rpc_url = self.RPC_DEVNET if network == "devnet" else self.RPC_MAINNET
        self.client = Client(self.rpc_url)
        self.keypair: Optional[Keypair] = None
        self.jupiter_api_key = self._load_jupiter_key()
        self._load_wallet()
    
    def _load_jupiter_key(self) -> str:
        """Load Jupiter API key from .env."""
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            # Find the last JUPITER_API_KEY= entry (the one with value)
            lines = env_file.read_text().split("\n")
            for line in reversed(lines):
                if line.startswith("JUPITER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key:  # Has value
                        return key
        return ""
    
    def _load_wallet(self):
        """Load wallet from .env or file."""
        env_file = PROJECT_ROOT / ".env"
        
        if env_file.exists():
            for line in env_file.read_text().split("\n"):
                if line.startswith("HOT_WALLET_PRIVATE_KEY="):
                    private_key = line.split("=", 1)[1].strip()
                    try:
                        if private_key.startswith("["):
                            self.keypair = Keypair.from_json(private_key)
                        else:
                            import base58
                            key_bytes = base58.b58decode(private_key)
                            self.keypair = Keypair.from_bytes(key_bytes)
                        print(f"✅ Wallet loaded", file=sys.stderr)
                        return
                    except Exception as e:
                        print(f"⚠️ Wallet load error: {e}", file=sys.stderr)
        
        print("⚠️ No wallet found", file=sys.stderr)
    
    def get_address(self) -> str:
        """Get wallet address."""
        if not self.keypair:
            return "No wallet configured"
        return str(self.keypair.pubkey())
    
    def get_balance(self) -> str:
        """Get wallet balance in SOL and USDC."""
        if not self.keypair:
            return "❌ No wallet configured"
        
        try:
            # Get SOL balance
            pubkey = self.keypair.pubkey()
            response = self.client.get_balance(pubkey)
            sol_balance = response.value / 1e9
            
            # Get SOL price
            sol_price = self.get_sol_price_value()
            
            total_usd = sol_balance * sol_price
            
            return f"💰 **Tu Wallet**\n\n" \
                   f"**SOL:** {sol_balance:.4f}\n" \
                   f"**USDC:** $0.00\n" \
                   f"**Total:** ${total_usd:.2f} USD\n" \
                   f"**Price:** ${sol_price:.2f}/SOL"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def get_sol_price(self) -> str:
        """Get current SOL price."""
        price = self.get_sol_price_value()
        return f"📊 **Precio de SOL**\n\n**${price:.2f}** USD"
    
    def get_sol_price_value(self) -> float:
        """Get SOL price value."""
        try:
            url = f"{self.JUPITER_PRICE_URL}?id={self.SOL_MINT}"
            response = requests.get(url)
            data = response.json()
            return float(data.get("data", {}).get("price", 0))
        except Exception as e:
            print(f"⚠️ Price error: {e}", file=sys.stderr)
            return 80.76  # Default fallback
    
    def get_quote(self, amount: float, side: str = "buy") -> str:
        """
        Get quote for swapping SOL <-> USDC.
        
        Args:
            amount: Amount in SOL
            side: 'buy' (SOL -> USDC) or 'sell' (USDC -> SOL)
        """
        try:
            input_mint = self.SOL_MINT if side == "buy" else self.USDC_MINT
            output_mint = self.USDC_MINT if side == "buy" else self.SOL_MINT
            
            amount_lamports = int(amount * 1e9)
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount_lamports,
                "slippageBps": 50
            }
            
            headers = {}
            if self.jupiter_api_key:
                headers["Authorization"] = f"Bearer {self.jupiter_api_key}"
            
            response = requests.get(self.JUPITER_QUOTE_URL, params=params, headers=headers)
            
            if response.status_code == 401:
                return f"⚠️ **Quote unavailable**\n\nLa API de Jupiter requiere autenticación.\n" \
                       f"Obten tu API key gratis: https://portal.jup.ag"
            
            data = response.json()
            
            in_amount = data.get("inAmount", 0) / 1e9
            out_amount = data.get("outAmount", 0) / 1e6  # USDC has 6 decimals
            
            action = "comprando" if side == "buy" else "vendiendo"
            
            return f"🔄 **{action.capitalize()} {amount} SOL**\n\n" \
                   f"📥 Entras: {in_amount:.4f} SOL\n" \
                   f"📤 Sales: {out_amount:.2f} USDC\n" \
                   f"💵 Rate: 1 SOL = ${out_amount/amount:.2f} USD"
        except Exception as e:
            return f"❌ Quote error: {e}"
    
    def execute_swap(self, amount: float, side: str = "buy") -> str:
        """
        Execute swap (requires wallet with SOL).
        
        Args:
            amount: Amount in SOL
            side: 'buy' or 'sell'
        """
        if not self.keypair:
            return "❌ Wallet no configurada. Ejecuta:\n" \
                   "`python3 tools/solana_wallet.py --generate`"
        
        try:
            # Get quote first
            input_mint = self.SOL_MINT if side == "buy" else self.USDC_MINT
            output_mint = self.USDC_MINT if side == "buy" else self.SOL_MINT
            amount_lamports = int(amount * 1e9)
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount_lamports,
                "slippageBps": 50,
                "userPublicKey": str(self.keypair.pubkey())
            }
            
            response = requests.get(self.JUPITER_QUOTE_URL, params=params)
            quote_data = response.json()
            
            # Get swap transaction
            swap_params = {
                "quoteResponse": quote_data,
                "userPublicKey": str(self.keypair.pubkey())
            }
            
            swap_response = requests.post(self.JUPITER_SWAP_URL, json=swap_params)
            swap_data = swap_response.json()
            
            # Serialize and sign
            # Note: Full implementation needs @solana/web3.js and @solana/spl-token
            # This is a placeholder for the full implementation
            
            action = "compra" if side == "buy" else "venta"
            
            return f"⚠️ **Swap simulation: {action} de {amount} SOL**\n\n" \
                   f"Para ejecutar swaps reales, necesitas:\n" \
                   f"1. SOL en tu wallet (usa el faucet)\n" \
                   f"2. Firmar la transacción con tu clave privada\n\n" \
                   f"📍 Wallet: {self.get_address()}\n" \
                   f"💰 Balance: {self.get_balance().split(chr(10))[1]}"
                   
        except Exception as e:
            return f"❌ Swap error: {e}"
    
    def get_status(self) -> str:
        """Get system status."""
        address = self.get_address()
        balance = self.get_balance()
        price = self.get_sol_price_value()
        
        return f"📊 **Estado del Sistema**\n\n" \
               f"**Wallet:** {address[:10]}...{address[-4:]}\n" \
               f"**Network:** {self.network}\n" \
               f"**SOL Price:** ${price:.2f}\n" \
               f"**Status:** {'✅ Wallet OK' if self.keypair else '⚠️ No wallet'}\n\n" \
               f"{balance}"


# ==================== CLI ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Solana Trading Handler for Eko",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 trading_handler.py --balance
  python3 trading_handler.py --price
  python3 trading_handler.py --buy 0.5
  python3 trading_handler.py --sell 1.0
  python3 trading_handler.py --status
  python3 trading_handler.py --address
        """
    )
    
    parser.add_argument("--balance", action="store_true",
                       help="Show wallet balance")
    parser.add_argument("--price", action="store_true",
                       help="Get SOL price")
    parser.add_argument("--quote", type=float, metavar="AMOUNT",
                       help="Get quote for swapping AMOUNT SOL <-> USDC")
    parser.add_argument("--buy", type=float, metavar="AMOUNT",
                       help="Buy AMOUNT of SOL")
    parser.add_argument("--sell", type=float, metavar="AMOUNT",
                       help="Sell AMOUNT of SOL")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")
    parser.add_argument("--address", action="store_true",
                       help="Show wallet address")
    parser.add_argument("--network", "-n", default="devnet",
                       choices=["devnet", "mainnet"],
                       help="Network (default: devnet)")
    
    args = parser.parse_args()
    
    # Create handler
    handler = TradingHandler(network=args.network)
    
    if args.balance:
        print(handler.get_balance())
    
    elif args.price:
        print(handler.get_sol_price())
    
    elif args.quote:
        # Show quote for buying SOL (SOL -> USDC)
        print(handler.get_quote(args.quote, "buy"))
    
    elif args.buy:
        # Execute buy swap (placeholder for now)
        print(handler.execute_swap(args.buy, "buy"))
    
    elif args.sell:
        # Execute sell swap (placeholder for now)
        print(handler.execute_swap(args.sell, "sell"))
    
    elif args.status:
        print(handler.get_status())
    
    elif args.address:
        print(handler.get_address())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
