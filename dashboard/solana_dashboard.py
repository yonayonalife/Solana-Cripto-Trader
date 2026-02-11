#!/usr/bin/env python3
"""
Solana Dashboard - Streamlit Interface
======================================
Web dashboard for monitoring and controlling the trading bot.

Based on: interface.py from Coinbase Cripto Trader

Features:
- Real-time worker status
- Strategy performance metrics
- Backtest results visualization
- Trade history
- System controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Jupiter Solana Trading Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title("🚀 Jupiter Solana Trading Bot")
st.markdown("---")


# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Network selector
    network = st.selectbox(
        "Network",
        ["devnet", "testnet", "mainnet"],
        index=0
    )
    
    # Risk level
    risk_level = st.select_slider(
        "Risk Level",
        options=["LOW", "MEDIUM", "HIGH"],
        value="MEDIUM"
    )
    
    st.markdown("---")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (s)", 5, 60, 10)
    
    st.markdown("---")
    
    # Connection status
    st.subheader("🔗 Connection")
    
    if st.button("Test RPC Connection"):
        with st.spinner("Testing connection..."):
            try:
                # Placeholder for actual test
                st.success("✅ Connected!")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ============================================================================
# TAB 1: DASHBOARD OVERVIEW
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "👷 Workers", 
    "📈 Strategies",
    "🔄 Swap",
    "🎮 Control",
    "📋 Logs"
])


with tab1:
    # Get real wallet data
    import httpx
    from solana.rpc.api import Client
    from solders.pubkey import Pubkey
    
    WALLET_ADDRESS = "65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3"
    
    # Get SOL balance
    sol_balance = 0.0
    sol_price = 0.0
    try:
        rpc = Client('https://api.devnet.solana.com')
        pubkey = Pubkey.from_string(WALLET_ADDRESS)
        resp = rpc.get_balance(pubkey)
        sol_balance = resp.value / 1_000_000_000
        
        # Get SOL price
        url = 'https://lite-api.jup.ag/price/v3?ids=So11111111111111111111111111111111111111112'
        resp = httpx.get(url, timeout=10)
        data = resp.json()
        sol_price = data.get('So11111111111111111111111111111111111111112', {}).get('usdPrice', 0)
    except Exception as e:
        st.error(f"Error connecting to Solana: {e}")
    
    portfolio_value = sol_balance * sol_price
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Portfolio Value",
            f"${portfolio_value:.2f}",
            delta=f"{sol_balance:.4f} SOL",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "📈 SOL Price",
            f"${sol_price:.2f}",
            delta="Live",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "🎯 SOL Balance",
            f"{sol_balance:.4f}",
            delta="Native SOL",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "🔥 Active Positions",
            "1",  # Solo SOL nativo por ahora
            delta="Native Only",
            delta_color="off"
        )
    
    st.markdown("---")
    
    # Wallet Info
    with st.expander("📍 Wallet Address", expanded=True):
        st.code(WALLET_ADDRESS, language="text")
        st.caption("Esta wallet tiene SOL nativo. Para tokens SPL, necesitas depósitos adicionales.")
    
    # SOL Native Balance Display
    col_sol, col_chart2 = st.columns([1, 2])
    
    with col_sol:
        st.subheader("💰 SOL Native")
        st.metric("Balance", f"{sol_balance:.4f} SOL")
        st.metric("Value (USD)", f"${portfolio_value:.2f}")
        st.metric("Price", f"${sol_price:.2f}")
    
    # Token Holdings (from Jupiter API)
    with col_chart2:
        st.subheader("📊 Token Holdings")
        
        # Try to get token holdings
        holdings_url = f'https://lite-api.jup.ag/ultra/v1/holdings/{WALLET_ADDRESS}'
        try:
            resp = httpx.get(holdings_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get('tokens', {})
                if tokens:
                    # Create table
                    holdings_data = []
                    for mint, info in tokens.items():
                        amount = float(info.get('uiAmount', 0))
                        if amount > 0:
                            holdings_data.append({
                                'Token': mint[:12] + '...',
                                'Amount': amount,
                                'USD Value': float(info.get('usdValue', 0))
                            })
                    if holdings_data:
                        df = pd.DataFrame(holdings_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No SPL tokens found. Deposit tokens to see them here.")
                else:
                    st.info("💡 No SPL tokens found. Esta wallet solo tiene SOL nativo.")
            else:
                st.info("💡 No SPL tokens found. Esta wallet solo tiene SOL nativo.")
        except Exception as e:
            st.info(f"💡 Esta wallet solo tiene SOL nativo. ({str(e)[:50]})")
    
    st.markdown("---")
    
    # Historical chart placeholder
    st.subheader("📈 Portfolio Performance")
    st.info("📊 Gráfico de rendimiento coming soon...")
    st.line_chart([1, 2, 3, 5, 8, 13])  # Placeholder
    
    with col_chart2:
        st.subheader("📈 P&L Chart")
        
        # Generate sample data
        dates = pd.date_range(start="2024-01-01", periods=30)
        prices = 100 + np.cumsum(np.random.randn(30) * 2)
        
        df = pd.DataFrame({"Date": dates, "Value": prices})
        df.set_index("Date", inplace=True)
        
        st.line_chart(df)


# ============================================================================
# TAB 2: WORKERS STATUS
# ============================================================================

with tab2:
    st.subheader("👷 Worker Status")
    
    # Sample worker data
    workers = [
        {"id": "MacBookPro_W1", "status": "🟢", "cpu": 45, "memory": 60, "tasks": 12},
        {"id": "MacBookPro_W2", "status": "🟢", "cpu": 38, "memory": 55, "tasks": 10},
        {"id": "MacBookPro_W3", "status": "🟡", "cpu": 72, "memory": 80, "tasks": 8},
        {"id": "LinuxROG_W1", "status": "🟢", "cpu": 55, "memory": 45, "tasks": 15},
        {"id": "LinuxROG_W2", "status": "🔴", "cpu": 0, "memory": 0, "tasks": 0},
    ]
    
    # Workers table
    df_workers = pd.DataFrame(workers)
    
    # Format for display
    def color_status(status):
        if "🟢" in status:
            return "green"
        elif "🟡" in status:
            return "orange"
        else:
            return "red"
    
    st.dataframe(
        df_workers,
        column_config={
            "status": st.column_config.TextColumn("Status"),
            "cpu": st.column_config.ProgressColumn("CPU %", min_value=0, max_value=100),
            "memory": st.column_config.ProgressColumn("Memory %", min_value=0, max_value=100),
        },
        width="stretch"
    )
    
    # Worker controls
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        st.button("🔄 Restart All Workers")
    
    with col_w2:
        st.button("📊 Force Sync")


# ============================================================================
# TAB 3: STRATEGIES
# ============================================================================

with tab3:
    st.subheader("📈 Strategy Performance")
    
    # Sample strategy results
    strategies = [
        {"name": "RSI_Breve", "pnl": 12.5, "trades": 45, "win_rate": 68.2, "sharpe": 1.85},
        {"name": "SMA_Crossover", "pnl": 8.3, "trades": 32, "win_rate": 72.1, "sharpe": 1.52},
        {"name": "EMA_Strategy", "pnl": -2.1, "trades": 28, "win_rate": 45.3, "sharpe": 0.42},
        {"name": "Volume_Breakout", "pnl": 15.7, "trades": 18, "win_rate": 78.9, "sharpe": 2.15},
        {"name": "Momentum", "pnl": 5.4, "trades": 55, "win_rate": 61.2, "sharpe": 1.25},
    ]
    
    df_strategies = pd.DataFrame(strategies)
    
    # Sort by PnL
    df_strategies = df_strategies.sort_values("pnl", ascending=False)
    
    # Display strategies table
    st.dataframe(
        df_strategies,
        width="stretch"
    )
    
    # Best strategy details
    st.markdown("---")
    st.subheader("🏆 Best Strategy Details")
    
    best_strategy = df_strategies.iloc[0]
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric("Strategy", best_strategy["name"])
    with col_s2:
        st.metric("PnL", f"{best_strategy['pnl']:.1f}%")
    with col_s3:
        st.metric("Win Rate", f"{best_strategy['win_rate']:.1f}%")
    with col_s4:
        st.metric("Sharpe Ratio", f"{best_strategy['sharpe']:.2f}")
    
    # Strategy genome
    with st.expander("🧬 Strategy Genome"):
        st.code(json.dumps({
            "sl_pct": 0.03,
            "tp_pct": 0.06,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        }, indent=2))


# ============================================================================
# TAB 4: MANUAL SWAP
# ============================================================================

with tab4:
    st.subheader("🔄 Manual Swap (DEVNET)")
    
        # All popular tokens on Solana
    ALL_TOKENS = {
        # Top by Market Cap
        "SOL": ("So11111111111111111111111111111111111111112", 9),
        "USDC": ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 6),
        "USDT": ("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYW", 6),
        
        # DeFi / Infrastructure
        "JUP": ("JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2", 6),
        "RAY": ("4jNcW4C4m6TnB8e9pWpB7Yq4Yq4Yq4Yq4Yq4Yq4Yq4Y", 6),
        "MNGO": ("MangoV3Maint8S4XKJB7hqFDJWsp8PxcYqBtkBxF9V", 6),
        "SRM": ("SRMuApVNdxXokE5vYV5M4kE8V9nV5M4kE8V9nV5M4k", 6),
        "ORCA": ("orcaEKTmK8nd7G6q8f4Yq4Yq4Yq4Yq4Yq4Yq4Yq4Y", 6),
        
        # Memecoins Populares
        "BONK": ("DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP", 5),
        "WIF": ("EKpQGSJtjMFqKZ9KQanSqWJcNSPWfqHYJQD7i阜eLJ", 6),
        "WEN": ("WENWENv2ykuwsLVnK4KbYQaN9UJqr4Yz7X6gYVfY8X", 5),
        "POPCAT": ("7GCihgDB8dfS1XbY9Hb86h7V9Y4vNPQp1dz1Yyj4FU", 9),
        "MEW": ("MEW1gQWJ3nEXg2uoERb2YbR1w6G3Yq4Yq4Yq4Yq4Yq4", 5),
        "FLOKI": ("FLuxi2vNLG4JLc1K8p9w9Yq4Yq4Yq4Yq4Yq4Yq4Yq", 9),
        
        # AI / Gaming
        "PYTH": ("HZ1JovNiBEgZ1W7E2hKQzF8Tz3G6fZ6K3jKGn1c3bY7V", 6),
        "ATLAS": ("ATLASXmbVVx9CPvCxgqHXa1CU9G4yr7V6Wf9Y7Yq1z", 8),
        "STARL": ("STARLr4x1oZvi1b7Y9gX7ZVG3V7Zq7Z7Zq7Zq7Zq7Zq", 8),
        "COPE": ("CPEz5niaEVfD3vKKB7xHVDqs1K1L4V7K7K7K7K7K7K7K7", 6),
        "HNT": ("HNT Token address here", 8),
        "AUDIO": ("AUDIO Token address here", 8),
        "MNDE": ("MNDE Token address here", 6),
    }
    
    # Tokens con precio en tiempo real
    PRICE_TOKENS = ["SOL", "USDC", "USDT", "JUP", "BONK", "WIF", "PYTH", "WEN", "MNGO"]
    
    col_swap1, col_swap2 = st.columns(2)
    
    with col_swap1:
        st.markdown("### 💸 Sell")
        sell_token = st.selectbox("Token", list(ALL_TOKENS.keys()), key="sell")
        sell_amount = st.number_input("Amount", min_value=0.0, value=0.1, step=0.1, key="sell_amount")
    
    with col_swap2:
        st.markdown("### 🛒 Buy")
        buy_token = st.selectbox("Token", list(ALL_TOKENS.keys()), key="buy")
        st.text_input("Est. Output", value="Click 'Get Quote' below", disabled=True, key="output")
    
    st.markdown("---")
    
    if st.button("📊 Get Quote", type="primary"):
        with st.spinner("Getting quote from Jupiter..."):
            try:
                import httpx
                
                tokens = {k: v[0] for k, v in ALL_TOKENS.items()}
                decimals = {k: v[1] for k, v in ALL_TOKENS.items()}
                
                input_mint = tokens[sell_token]
                output_mint = tokens[buy_token]
                
                # Convert to lamports
                amount = int(sell_amount * (10 ** decimals[sell_token]))
                
                url = f"https://lite-api.jup.ag/ultra/v1/order?inputMint={input_mint}&outputMint={output_mint}&amount={amount}"
                resp = httpx.get(url, timeout=30.0)
                data = resp.json()
                
                if "outAmount" in data:
                    out_amount = int(data["outAmount"])
                    output_val = out_amount / (10 ** decimals[buy_token])
                    st.success(f"💰 Output: {output_val:.6f} {buy_token}")
                    st.info(f"📊 Price Impact: {data.get('priceImpact', 'N/A')}%")
                    st.info(f"⏱️ Route Time: {data.get('totalTime', 'N/A')}ms")
                    
                    if st.button("🚀 EXECUTE SWAP", type="primary"):
                        st.error("⚠️ Swap requires private key setup. Use CLI.")
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Quick prices - more tokens
    st.subheader("📈 Current Prices")
    
    try:
        import httpx
        
        # Get prices for popular tokens
        mints = [ALL_TOKENS[t][0] for t in PRICE_TOKENS[:6]]
        ids = "%2C".join(mints)
        
        url = f"https://lite-api.jup.ag/price/v3?ids={ids}"
        resp = httpx.get(url, timeout=30.0)
        data = resp.json()
        
        # Create price display
        cols = st.columns(3)
        i = 0
        for token in PRICE_TOKENS[:6]:
            addr = ALL_TOKENS[token][0]
            if addr in data:
                price = data[addr].get("usdPrice", 0)
                with cols[i % 3]:
                    st.metric(token, f"${price:.4f}")
                i += 1
    except Exception as e:
        st.error(f"Error: {e}")
        for i, (addr, info) in enumerate(data.items()):
            symbol = addr[:8]
            price = info.get("usdPrice", 0)
            with [col_p1, col_p2, col_p3][i]:
                st.metric(symbol, f"${price:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================================
# TAB 5: CONTROL PANEL
# ============================================================================

with tab5:
    st.subheader("🎮 System Control")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("🚀 Start/Stop")
        
        if st.button("▶️ Start All Workers", type="primary"):
            st.success("Starting all workers...")
        
        if st.button("⏹️ Stop All Workers"):
            st.error("Stopping all workers...")
        
        if st.button("🔄 Restart Coordinator"):
            st.warning("Restarting coordinator...")
    
    with col_c2:
        st.subheader("📝 Actions")
        
        action = st.selectbox(
            "Select Action",
            [
                "Create Work Unit",
                "Run Single Backtest",
                "Export Results",
                "Clear Database"
            ]
        )
        
        if action == "Create Work Unit":
            st.text_input("Population Size", value="20")
            st.text_input("Generations", value="50")
            st.text_input("Risk Level", value="MEDIUM")
            st.button("🚀 Create Work Unit")
        
        elif action == "Run Single Backtest":
            st.text_input("Strategy Name", value="test_strategy")
            st.button("🏃 Run Backtest")
        
        elif action == "Export Results":
            format_choice = st.selectbox("Format", ["JSON", "CSV", "Excel"])
            st.button("📥 Export")
        
        elif action == "Clear Database":
            st.warning("⚠️ This will delete all data!")
            st.button("🗑️ Clear All Data", type="primary")
    
    st.markdown("---")
    
    # Configuration
    st.subheader("⚙️ Parameters")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.number_input("Max Position %", min_value=1, max_value=100, value=10)
    
    with col_p2:
        st.number_input("Stop Loss %", min_value=1, max_value=50, value=3)
    
    with col_p3:
        st.number_input("Take Profit %", min_value=1, max_value=100, value=6)


# ============================================================================
# TAB 5: LOGS
# ============================================================================

with tab6:
    st.subheader("📋 System Logs")
    
    # Log level filter
    log_filter = st.selectbox(
        "Filter by Level",
        ["ALL", "INFO", "WARNING", "ERROR"]
    )
    
    # Sample logs
    logs = [
        {"time": "2024-02-09 21:30:15", "level": "INFO", "message": "Worker W1: Received work unit #42"},
        {"time": "2024-02-09 21:30:12", "level": "INFO", "message": "Backtest completed: PnL=+5.23%"},
        {"time": "2024-02-09 21:30:10", "level": "WARNING", "message": "High latency detected: 234ms"},
        {"time": "2024-02-09 21:30:08", "level": "ERROR", "message": "Worker W2: Connection timeout"},
        {"time": "2024-02-09 21:30:05", "level": "INFO", "message": "Coordinator: Work unit #41 completed"},
    ]
    
    # Filter logs
    if log_filter != "ALL":
        logs = [l for l in logs if l["level"] == log_filter]
    
    # Display logs
    for log in logs:
        color = {
            "INFO": "🟢",
            "WARNING": "🟡",
            "ERROR": "🔴"
        }.get(log["level"], "⚪")
        
        st.text(f"{log['time']} {color} [{log['level']}] {log['message']}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🚀 Jupiter Solana Trading Bot | 
        Powered by MiniMax M2.1 & OpenClaw |
        v1.0.0
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Solana Dashboard...")
    print("Access at: http://localhost:8501")
    print("=" * 60)
