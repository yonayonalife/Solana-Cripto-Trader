#!/usr/bin/env python3
"""
ML Signal Generator
==================
Machine Learning-based trading signals using simple models.

Features:
- RSI-based signals
- Moving Average crossover
- Momentum scoring
- Sentiment analysis (simple)

Based on research from similar projects.

Usage:
    from ml_signals import MLSignalGenerator
    signals = generator.generate(data)
"""

import json
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketData:
    """Market data for ML analysis."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    timestamp: str


@dataclass
class Signal:
    """ML-generated signal."""
    symbol: str
    direction: str
    confidence: float
    reason: str
    indicators: Dict[str, float]


class TechnicalIndicators:
    """Calculate technical indicators."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 70.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0

        alpha = 2 / (period + 1)
        ema = prices[-1]
        for price in reversed(prices[:-1]):
            ema = alpha * price + (1 - alpha) * ema
        return ema

    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 10) -> float:
        """Calculate momentum."""
        if len(prices) < period:
            return 100.0

        current = prices[-1]
        past = prices[-period]
        return (current / past) * 100

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 14) -> float:
        """Calculate volatility (std dev)."""
        if len(prices) < period:
            return 0.02

        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)


class MLSignalGenerator:
    """
    ML-based signal generator.
    Uses ensemble of simple models for robustness.
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.price_history: Dict[str, List[float]] = {}

    def add_price(self, symbol: str, price: float):
        """Add price to history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

        # Keep only last 50 prices
        if len(self.price_history[symbol]) > 50:
            self.price_history[symbol] = self.price_history[symbol][-50:]

    def generate(self, data: MarketData) -> Optional[Signal]:
        """Generate ML signal for a token."""
        symbol = data.symbol
        price = data.price
        change = data.change_24h

        # Add price to history
        self.add_price(symbol, price)

        # Get price history
        history = self.price_history.get(symbol, [price])

        # Calculate indicators
        rsi = self.indicators.calculate_rsi(history)
        ema_9 = self.indicators.calculate_ema(history, 9)
        ema_21 = self.indicators.calculate_ema(history, 21)
        momentum = self.indicators.calculate_momentum(history)
        volatility = self.indicators.calculate_volatility(history)

        # Ensemble scoring
        scores = {
            "rsi": 0.0,
            "crossover": 0.0,
            "momentum": 0.0,
            "trend": 0.0
        }

        reasons = []

        # RSI scoring (0-100)
        if rsi < 30:
            scores["rsi"] = 0.8  # Oversold - BUY
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            scores["rsi"] = -0.8  # Overbought - SELL
            reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            scores["rsi"] = (rsi - 50) / 25  # Neutral zone

        # EMA crossover scoring
        if ema_9 > ema_21:
            scores["crossover"] = 0.5
            reasons.append("EMA 9 > 21 (bullish)")
        else:
            scores["crossover"] = -0.5
            reasons.append("EMA 9 < 21 (bearish)")

        # Momentum scoring
        if momentum > 105:
            scores["momentum"] = 0.6
            reasons.append(f"Strong momentum ({momentum:.1f}%)")
        elif momentum < 95:
            scores["momentum"] = -0.6
            reasons.append(f"Weak momentum ({momentum:.1f}%)")
        else:
            scores["momentum"] = (momentum - 100) / 10

        # 24h change trend
        if change > 5:
            scores["trend"] = 0.5
            reasons.append(f"Strong 24h ({change:+.1f}%)")
        elif change < -5:
            scores["trend"] = -0.5
            reasons.append(f"Weak 24h ({change:+.1f}%)")
        else:
            scores["trend"] = change / 20

        # Weighted ensemble
        weights = {
            "rsi": 0.30,
            "crossover": 0.25,
            "momentum": 0.25,
            "trend": 0.20
        }

        total_score = sum(scores[k] * weights[k] for k in scores)

        # Adjust for volatility (high volatility = lower confidence)
        vol_penalty = min(volatility * 5, 0.3)
        confidence = abs(total_score) - vol_penalty
        confidence = max(0, min(0.95, confidence))

        # Determine direction
        if total_score > 0.2:
            direction = "BUY"
        elif total_score < -0.2:
            direction = "SELL"
        else:
            return None  # No clear signal

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=" | ".join(reasons[:3]),
            indicators={
                "rsi": rsi,
                "ema_9": ema_9,
                "ema_21": ema_21,
                "momentum": momentum,
                "volatility": volatility
            }
        )

    def batch_generate(self, data_list: List[MarketData]) -> List[Signal]:
        """Generate signals for multiple tokens."""
        signals = []
        for data in data_list:
            signal = self.generate(data)
            if signal:
                signals.append(signal)
        return signals


async def test_ml_generator():
    """Test the ML generator."""
    import httpx

    print("\n🧪 TESTING ML SIGNAL GENERATOR")
    print("="*50)

    # Test tokens
    tokens = {
        "SOL": "So11111111111111111111111111111111111111112",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    }

    generator = MLSignalGenerator()

    # Add fake history for each token
    for symbol in tokens:
        base_price = 87.0 if symbol == "SOL" else 0.001
        for i in range(30):
            variation = np.random.normal(0, 0.02)
            generator.add_price(symbol, base_price * (1 + variation))

    # Fetch real prices and generate signals
    async with httpx.AsyncClient() as client:
        for symbol, address in tokens.items():
            try:
                resp = await client.get(
                    f"https://api.dexscreener.com/latest/dex/tokens/{address}",
                    timeout=10
                )
                data = resp.json()
                pairs = data.get("pairs", [])

                if pairs:
                    pair = pairs[0]
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(pair.get("priceUsd", 0)),
                        change_24h=float(pair.get("priceChange", {}).get("h24", 0)),
                        volume_24h=float(pair.get("volume", {}).get("h24", 0)),
                        timestamp=datetime.now().isoformat()
                    )

                    signal = generator.generate(market_data)

                    if signal:
                        emoji = "🟢" if signal.direction == "BUY" else "🔴"
                        print(f"\n{emoji} {symbol}: {signal.direction} ({signal.confidence:.0%})")
                        print(f"   Reason: {signal.reason}")
                        print(f"   RSI: {signal.indicators['rsi']:.1f}")
                        print(f"   Momentum: {signal.indicators['momentum']:.1f}%")
                    else:
                        print(f"\n⚪ {symbol}: No signal (neutral)")

            except Exception as e:
                print(f"Error {symbol}: {e}")

    print("\n" + "="*50)


if __name__ == "__main__":
    asyncio.run(test_ml_generator())
