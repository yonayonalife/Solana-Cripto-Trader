#!/usr/bin/env python3
"""
Simple Trading Dashboard
=====================
Lightweight dashboard without heavy dependencies.

Features:
- Clean HTML interface
- Auto-refresh every 10 seconds
- System metrics
- Trade history

Usage:
    python3 dashboard/simple_dashboard.py 8502
    Then open: http://localhost:8502
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
from pathlib import Path

# Configuration
STATE_FILE = Path(__file__).parent.parent / "unified_brain_state.json"
DB_FILE = Path(__file__).parent.parent / "db" / "unified_trading.db"
PORT = 8502


def get_state():
    """Get brain state."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def get_trades():
    """Get all trades."""
    if DB_FILE.exists():
        data = json.loads(DB_FILE.read_text())
        return data.get("trades", [])
    return []


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard."""

    def do_GET(self):
        """Serve dashboard page."""
        state = get_state()
        trades = get_trades()
        stats = state.get("stats", {})

        daily_pnl = stats.get("daily_pnl_pct", 0)
        progress = min(abs(daily_pnl) / 5 * 100, 100) if daily_pnl != 0 else 0
        pnl_color = "color:#00cc96" if daily_pnl >= 0 else "color:#ef553b"

        # Format trades
        trades_html = ""
        for trade in reversed(trades[-10:]):
            s = trade.get("symbol", "")
            d = trade.get("direction", "")
            p = trade.get("pnl_pct") or 0
            st = trade.get("status", "")
            dc = "background:#00cc96" if d == "BUY" else "background:#ef553b"
            pnc = "color:#00cc96" if p >= 0 else "color:#ef553b"
            trades_html += (
                f"<tr><td>{s}</td>"
                f'<td style="{dc};padding:3px 8px;border-radius:4px;color:#000;font-weight:bold">{d}</td>'
                f'<td style="{pnc};font-weight:bold">{p:+.2f}%</td>'
                f'<td style="color:#888">{st}</td></tr>'
            )

        if not trades_html:
            trades_html = '<tr><td colspan="4" style="text-align:center;color:#666">No trades yet</td></tr>'

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
<title>Eko Trading Dashboard</title>
<meta http-equiv="refresh" content="10">
<style>
body{{font-family:-apple-system,sans-serif;background:linear-gradient(135deg,#0a0a0a,#1a1a2e);color:#fff;margin:0;padding:20px}}
.header{{text-align:center;padding:30px;background:rgba(255,255,255,0.05);border-radius:15px;margin-bottom:20px}}
h1{{background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}}
.status{{display:inline-block;padding:5px 15px;background:#00cc96;border-radius:20px;font-size:12px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:15px;margin-bottom:20px}}
.metric{{background:rgba(255,255,255,0.05);border-radius:12px;padding:20px;text-align:center}}
.value{{font-size:28px;font-weight:bold}}
.label{{font-size:11px;color:#888;text-transform:uppercase}}
.section{{background:rgba(255,255,255,0.03);border-radius:15px;padding:20px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse}}
td{{padding:10px;border-bottom:1px solid rgba(255,255,255,0.05)}}
th{{text-align:left;color:#888;font-size:11px;text-transform:uppercase;padding:10px;border-bottom:1px solid rgba(255,255,255,0.1)}}
.modules{{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:10px}}
.module{{background:rgba(255,255,255,0.03);padding:10px;border-radius:8px;text-align:center;font-size:11px}}
.progress{{width:100%;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;margin:15px 0}}
.fill{{height:100%;background:linear-gradient(90deg,#00cc96,#00d4ff);border-radius:4px}}
</style>
</head>
<body>
<div class="header">
<h1>Eko Trading System</h1>
<span class="status"> ONLINE</span>
<p style="color:#666;margin-top:10px">Professional Trading Dashboard</p>
</div>

<div class="metrics">
<div class="metric"><div class="value">{len(trades)}</div><div class="label">Total Trades</div></div>
<div class="metric"><div class="value" style="{pnl_color}">{daily_pnl:+.2f}%</div><div class="label">Daily P&L</div></div>
<div class="metric"><div class="value">{stats.get("cycles", 0)}</div><div class="label">Cycles</div></div>
<div class="metric"><div class="value">{stats.get("trades_today", 0)}</div><div class="label">Today</div></div>
</div>

<div class="section">
<h3>Progress - Daily Target: +5%</h3>
<div class="progress"><div class="fill" style="width:{progress:.1f}%"></div></div>
<p style="text-align:center;color:#888;font-size:12px">{progress:.1f}% of daily target</p>
</div>

<div class="section">
<h3>System Modules</h3>
<div class="modules">
<div class="module"> WebSocket</div>
<div class="module"> Jito</div>
<div class="module"> Database</div>
<div class="module"> ML Signals</div>
<div class="module"> Redis</div>
<div class="module"> Scout</div>
<div class="module"> Optimizer</div>
</div>
</div>

<div class="section">
<h3>Recent Trades</h3>
<table>
<tr><th>Symbol</th><th>Dir</th><th>P&L</th><th>Status</th></tr>
{trades_html}
</table>
</div>

<p style="text-align:center;color:#666;font-size:12px;margin-top:20px">Last updated: {datetime.now().strftime("%H:%M:%S")}</p>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        pass


def run_dashboard(port=PORT):
    """Run the dashboard server."""
    print(f"\n Dashboard running at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    HTTPServer(("", port), DashboardHandler).serve_forever()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_dashboard(port)
