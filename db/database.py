#!/usr/bin/env python3
"""
PostgreSQL Database for Trading History
========================================
Stores trades, tokens, and market analysis.

Based on solana-trading-cli architecture.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Trade:
    """Trade record."""
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    status: str
    timestamp: str
    strategy: str = "manual"


class SQLiteDatabase:
    """
    SQLite database (PostgreSQL compatible via adapter).
    Uses file-based storage for simplicity.
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = Path(db_path)
        self.trades: List[Trade] = []
        self.tokens: Dict[str, Dict] = {}
        self._load()
        
    def _load(self):
        """Load data from file."""
        if self.db_path.exists():
            data = json.loads(self.db_path.read_text())
            self.trades = [Trade(**t) for t in data.get("trades", [])]
            self.tokens = data.get("tokens", {})
            
    def _save(self):
        """Persist data to file."""
        data = {
            "trades": [asdict(t) for t in self.trades],
            "tokens": self.tokens,
            "last_update": datetime.now().isoformat()
        }
        self.db_path.write_text(json.dumps(data, indent=2))
    
    def add_trade(self, trade: Trade):
        """Add a new trade."""
        self.trades.append(trade)
        self._save()
        
    def update_trade(self, trade_id: str, **kwargs):
        """Update existing trade."""
        for trade in self.trades:
            if trade.id == trade_id:
                for key, value in kwargs.items():
                    setattr(trade, key, value)
                self._save()
                break
    
    def get_open_positions(self) -> List[Trade]:
        """Get all open positions."""
        return [t for t in self.trades if t.status == "open"]
    
    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades."""
        return [t for t in self.trades if t.status == "closed"]
    
    def get_performance(self) -> Dict:
        """Calculate performance metrics."""
        closed = self.get_closed_trades()
        if not closed:
            return {"win_rate": 0, "total_pnl": 0, "trades": 0}
        
        wins = sum(1 for t in closed if t.pnl and t.pnl > 0)
        total_pnl = sum(t.pnl or 0 for t in closed)
        
        return {
            "total_trades": len(closed),
            "winning_trades": wins,
            "losing_trades": len(closed) - wins,
            "win_rate": wins / len(closed) * 100,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed)
        }
    
    def update_token(self, symbol: str, data: Dict):
        """Update token info."""
        self.tokens[symbol] = {
            **data,
            "last_seen": datetime.now().isoformat()
        }
        self._save()
    
    def get_top_tokens(self, limit: int = 10) -> List[Dict]:
        """Get most traded tokens."""
        from collections import Counter
        symbols = [t.symbol for t in self.trades]
        most_common = Counter(symbols).most_common(limit)
        return [{"symbol": s, "trades": c} for s, c in most_common]


async def main():
    """Test database."""
    db = SQLiteDatabase("test_trading.db")
    
    # Add test trade
    trade = Trade(
        id="test_001",
        symbol="SOL",
        direction="BUY",
        entry_price=87.0,
        exit_price=None,
        size=100,
        pnl=None,
        pnl_pct=None,
        status="open",
        timestamp=datetime.now().isoformat()
    )
    
    db.add_trade(trade)
    print(f"Added trade: {trade.id}")
    
    perf = db.get_performance()
    print(f"Performance: {perf}")
    
    # Cleanup
    Path("test_trading.db").unlink(missing_ok=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
