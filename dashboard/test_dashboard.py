#!/usr/bin/env python3
"""Simple test dashboard"""
import streamlit as st

st.title("🦞 Eko - Trading Dashboard")
st.write("✅ Dashboard working!")

col1, col2, col3 = st.columns(3)
col1.metric("SOL Price", "$80.76", "+0.5%")
col2.metric("Wallet", "0.0 SOL", "0.00")
col3.metric("Status", "Ready", "✅")

st.write("---")
st.write("🚀 Multi-Agent Trading System")
