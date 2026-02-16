#!/usr/bin/env python3
"""
Multi-Agent Trading System for Solana - WITH SMART SCANNER
"""
import asyncio
import json
import random
import requests
from pathlib import Path
from datetime import datetime
from solana.rpc.api import Client
from solders.pubkey import Pubkey

WALLET_ADDRESS = "H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a"
RPC_URL = "https://api.devnet.solana.com"

STATE_FILE = Path("~/.config/solana-jupiter-bot/multi_agent_state.json").expanduser()

# Base tokens + new opportunities
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW",
}

# Additional tokens to discover
EXTRA_TOKENS = {}

class MarketScanner:
    """Smart scanner that finds new opportunities"""
    
    def __init__(self):
        self.name = "Market Scanner"
        self.discovered_tokens = {}
        self.scan_count = 0
    
    def get_solana_new_tokens(self):
        """Scan for new Solana tokens"""
        new_tokens = {}
        try:
            # Use Birdeye API for trending tokens
            url = "https://public-api.birdeye.so/defi/v2/tokenlist?sort_by=volume24h&sort_type=desc&limit=50"
            headers = {"x-birdeye-api-key": "demo"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for token in data.get("data", [])[:20]:
                    addr = token.get("address", "")
                    if addr and len(addr) > 30:
                        symbol = token.get("symbol", "???")
                        price = float(token.get("price", 0))
                        if price > 0:
                            change = float(token.get("price_change24h", 0))
                            volume = float(token.get("volume24h", 0))
                            new_tokens[symbol] = {
                                "address": addr,
                                "price": price,
                                "change_24h": change,
                                "volume_24h": volume,
                                "liquidity": token.get("liquidity", 0)
                            }
        except Exception as e:
            print(f"Birdeye scan error: {e}")
        
        # Also check Raydium for new pools
        try:
            url = "https://api.raydium.io/v2/main/pools"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for pool in data.get("data", [])[:10]:
                    base = pool.get("baseToken", {})
                    if base:
                        sym = base.get("symbol", "")
                        addr = base.get("address", "")
                        if sym and addr and sym not in new_tokens:
                            price = float(pool.get("price", 0))
                            if price > 0:
                                new_tokens[sym] = {
                                    "address": addr,
                                    "price": price,
                                    "change_24h": 0,
                                    "volume_24h": float(pool.get("volume24h", 0)),
                                    "liquidity": float(pool.get("liquidity", 0))
                                }
        except Exception as e:
            print(f"Raydium scan error: {e}")
        
        return new_tokens
    
    def get_meme_coins(self):
        """Get trending meme coins"""
        meme_tokens = {}
        try:
            # Try DEX Screener
            url = "https://api.dexscreener.com/latest/dex/tokens/solana"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for pair in data.get("pairs", [])[:15]:
                    token = pair.get("baseToken", {})
                    if token:
                        addr = token.get("address", "")
                        sym = token.get("symbol", "")
                        if sym and addr:
                            price = float(pair.get("priceUsd", 0))
                            if price > 0:
                                change = float(pair.get("priceChange", {}).get("h24", 0))
                                volume = float(pair.get("volume", {}).get("h24", 0))
                                meme_tokens[sym] = {
                                    "address": addr,
                                    "price": price,
                                    "change_24h": change,
                                    "volume_24h": volume,
                                    "liquidity": pair.get("liquidity", {}).get("usd", 0)
                                }
        except Exception as e:
            print(f"DEX Screener error: {e}")
        
        return meme_tokens
    
    def scan(self):
        """Full market scan"""
        self.scan_count += 1
        prices = {}
        
        # Scan base tokens (Jupiter)
        try:
            ids = ",".join(TOKENS.values())
            resp = requests.get("https://lite-api.jup.ag/price/v3", params={"ids": ids}, timeout=10)
            data = resp.json()
            for sym, mint in TOKENS.items():
                if mint in data and isinstance(data[mint], dict):
                    prices[sym] = {"price": float(data[mint].get("usdPrice", 0)), 
                                   "change": float(data[mint].get("priceChange24h", 0)),
                                   "source": "jupiter"}
        except Exception as e:
            print(f"Jupiter error: {e}")
        
        # Fill with CoinGecko
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/simple/price", 
                params={"ids": "solana,bitcoin,tether,bonk,dogwifhat,pepe", "vs_currencies": "usd", "include_24hr_change": "true"}, timeout=10)
            data = resp.json()
            mapping = {"solana": "SOL", "bitcoin": "BTC", "tether": "USDT", "bonk": "BONK", "dogwifhat": "WIF", "pepe": "PEPE"}
            for cg, sym in mapping.items():
                if cg in data and isinstance(data[cg], dict) and sym not in prices:
                    prices[sym] = {"price": float(data[cg].get("usd", 0)), 
                                   "change": float(data[cg].get("usd_24h_change", 0)),
                                   "source": "coingecko"}
        except Exception as e:
            print(f"CoinGecko error: {e}")
        
        # Discover NEW opportunities (every 5 scans)
        if self.scan_count % 5 == 0:
            print("\n🌐 [Scanner] Discovering new tokens...")
            
            # Get meme coins
            meme_tokens = self.get_meme_coins()
            if meme_tokens:
                print(f"   Found {len(meme_tokens)} new tokens!")
                # Add top meme tokens to prices
                sorted_memes = sorted(meme_tokens.items(), key=lambda x: x[1].get("volume_24h", 0), reverse=True)[:5]
                for sym, info in sorted_memes:
                    prices[sym] = {"price": info["price"], 
                                   "change": info.get("change_24h", 0),
                                   "volume": info.get("volume_24h", 0),
                                   "liquidity": info.get("liquidity", 0),
                                   "source": "dex_screener"}
                    print(f"   🪙 {sym}: ${info['price']:.6f} (Vol: ${info.get('volume_24h', 0):,.0f})")
            
            self.discovered_tokens = meme_tokens
        
        return prices

class Analyst:
    def analyze(self, prices):
        opps = []
        for sym, d in prices.items():
            if not isinstance(d, dict):
                continue
            ch = d.get("change", 0)
            vol = d.get("volume", 0)
            
            # Strong move + high volume = opportunity
            if ch < -10:
                opps.append({"symbol": sym, "action": "BUY", "strength": abs(ch), "reason": f"Deep dip {ch:+.1f}%", "volume": vol})
            elif ch < -7 and vol > 100000:
                opps.append({"symbol": sym, "action": "BUY", "strength": abs(ch) + 5, "reason": f"Dip+Vol {ch:+.1f}%", "volume": vol})
            elif ch > 15 and vol > 100000:
                opps.append({"symbol": sym, "action": "SELL", "strength": ch, "reason": f"Momentum {ch:+.1f}%", "volume": vol})
        
        return sorted(opps, key=lambda x: x.get("strength", 0), reverse=True)

class RiskManager:
    def __init__(self):
        self.max_pos = 3
        self.tp = 5.0
        self.sl = 3.0
        self.min_volume = 10000  # Min $10k volume
    
    def validate_entry(self, opp, state):
        if len(state.get("positions", {})) >= self.max_pos:
            return False, "Max positions"
        if opp["symbol"] in state.get("positions", {}):
            return False, "Already in position"
        
        # Check volume for new tokens
        vol = opp.get("volume", 0)
        if vol > 0 and vol < self.min_volume:
            return False, f"Low volume ${vol:,.0f}"
        
        return True, "Approved"
    
    def check_exits(self, positions, prices):
        exits = []
        for sym, data in positions.items():
            if not isinstance(data, dict):
                continue
            if sym not in prices:
                continue
            price_info = prices[sym]
            if not isinstance(price_info, dict):
                continue
            current = price_info.get("price", 0)
            entry = data.get("entry_price", 0)
            amt = data.get("amount", 0)
            if entry > 0 and amt > 0 and current > 0:
                pnl = ((current - entry) / entry) * 100
                if pnl >= self.tp:
                    exits.append({"symbol": sym, "action": "TAKE_PROFIT", "reason": f"+{pnl:.1f}%", "amount": amt, "price": current, "pnl": pnl})
                elif pnl <= -self.sl:
                    exits.append({"symbol": sym, "action": "STOP_LOSS", "reason": f"{pnl:.1f}%", "amount": amt, "price": current, "pnl": pnl})
        return exits

class Trader:
    def __init__(self):
        self.client = Client(RPC_URL)
        self.wallet = Pubkey.from_string(WALLET_ADDRESS)
    
    def get_wallet(self):
        return self.client.get_balance(self.wallet).value / 1e9
    
    def execute_entry(self, sym, price, state):
        amount = state["capital_usd"] * 0.10 / price
        cost = state["capital_usd"] * 0.10
        if sym not in state["positions"]:
            state["positions"][sym] = {"amount": 0, "entry_price": price}
        state["positions"][sym]["amount"] += amount
        state["positions"][sym]["entry_price"] = price
        state["capital_usd"] -= cost
        state["trades"].append({"time": datetime.now().isoformat(), "action": "BUY", "symbol": sym, "price": price, "amount": amount, "cost": cost})
        return True
    
    def execute_exit(self, exit_info, state):
        sym = exit_info["symbol"]
        amt = exit_info["amount"]
        price = exit_info["price"]
        proceeds = amt * price
        state["capital_usd"] += proceeds
        state["trades"].append({"time": datetime.now().isoformat(), "action": exit_info["action"], "symbol": sym, "price": price, "proceeds": proceeds, "pnl": exit_info["pnl"]})
        del state["positions"][sym]
        return True

class StrategyGenerator:
    def generate(self):
        return {"id": f"S{random.randint(100,999)}", "name": random.choice(["Momentum", "Mean Rev", "Trend", "Dip Hunter"]), "created": datetime.now().isoformat()}

class Orchestrator:
    def __init__(self):
        self.scanner = MarketScanner()
        self.analyst = Analyst()
        self.risk = RiskManager()
        self.trader = Trader()
        self.strat_gen = StrategyGenerator()
        self.load_state()
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.state = json.load(f)
        else:
            self.state = {"capital_usd": 500, "positions": {}, "trades": [], "strategies": []}
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    async def cycle(self, n):
        print(f"\n{'='*60}\nCYCLE {n} - {datetime.now().strftime('%H:%M:%S')}")
        
        prices = self.scanner.scan()
        opps = self.analyst.analyze(prices)
        
        # Check exits first
        exits = self.risk.check_exits(self.state["positions"], prices)
        for ex in exits:
            self.trader.execute_exit(ex, self.state)
            emoji = "🎯" if ex["action"] == "TAKE_PROFIT" else "🛑"
            print(f"   {emoji} {ex['action']} {ex['symbol']}: {ex['reason']}")
        
        # Execute entries
        if not exits and opps:
            for o in opps:
                if o["action"] == "BUY":
                    ok, msg = self.risk.validate_entry(o, self.state)
                    price_info = prices.get(o["symbol"])
                    if ok and price_info and isinstance(price_info, dict):
                        self.trader.execute_entry(o["symbol"], price_info.get("price", 0), self.state)
                        print(f"   ✅ BUY {o['symbol']} @ ${price_info.get('price', 0):.6f} (Vol: ${o.get('volume', 0):,.0f})")
                        break
        
        # Strategy
        if n % 10 == 0:
            s = self.strat_gen.generate()
            self.state.setdefault("strategies", []).append(s)
            print(f"\n🧪 Strategy: {s['name']}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"💰 Capital: ${self.state['capital_usd']:.2f}")
        
        total_pos = 0
        for sym, d in self.state["positions"].items():
            if not isinstance(d, dict):
                continue
            cur = prices.get(sym, {}).get("price", 0) if isinstance(prices.get(sym), dict) else 0
            val = d.get("amount", 0) * cur
            pnl = ((cur - d.get("entry_price", 0)) / d.get("entry_price", 1) * 100) if d.get("entry_price", 0) > 0 else 0
            print(f"   📈 {sym}: {'🟢' if pnl>=0 else '🔴'} {pnl:+.1f}%")
            total_pos += val
        
        total = self.state["capital_usd"] + total_pos
        print(f"💎 Total: ${total:.2f} | P&L: ${total-500:+.2f}")
        print(f"🔄 Trades: {len(self.state['trades'])}")
        
        self.save_state()
    
    async def run(self):
        print("="*60)
        print("MULTI-AGENT SOLANA TRADING (SMART SCANNER)")
        print("="*60)
        
        n = 0
        while True:
            n += 1
            try:
                await self.cycle(n)
            except Exception as e:
                print(f"❌ {e}")
            await asyncio.sleep(60)

asyncio.run(Orchestrator().run())
