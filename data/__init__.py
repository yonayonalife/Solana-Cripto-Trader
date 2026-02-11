# Data Module - Historical price data for backtesting
from data.historical_data import (
    HistoricalDataManager,
    DexScreenerClient,
    HeliusClient,
    Candle,
    DataConfig,
    get_real_price,
    get_historical_data
)

# Alias for easier importing
DataManager = HistoricalDataManager

__all__ = [
    "HistoricalDataManager",
    "DataManager",  # Alias
    "DexScreenerClient",
    "HeliusClient",
    "Candle",
    "DataConfig",
    "get_real_price",
    "get_historical_data"
]
