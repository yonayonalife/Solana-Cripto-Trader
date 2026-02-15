#!/bin/bash
# Persistent Dashboard Runner
# Restarts automatically if killed

while true; do
    echo "$(date): Starting dashboard..."
    python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
from pathlib import Path

HTML = '''<html><head><meta charset=UTF-8><meta http-equiv=refresh content=15><title>Eko Dashboard</title><style>*{margin:0;padding:0}body{font-family:Segoe UI,sans-serif;background:#0a0a0a;color:#fff;padding:20px}.header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:15px;margin-bottom:20px;display:flex;justify-content:space-between}.logo{font-size:22px;font-weight:bold;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.status{background:#00cc96;color:#000;padding:5px 15px;border-radius:20px;font-size:12px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-bottom:20px}.metric{background:#1e1e1e;padding:20px;border-radius:12px;text-align:center}.value{font-size:28px;font-weight:bold}.label{font-size:11px;color:#888;text-transform:uppercase}.positive{color:#00cc96}.negative{color:#ef553b}.section{background:#1a1a2e;padding:20px;border-radius:15px;margin-bottom:20px}h3{margin-bottom:15px;border-bottom:1px solid #333}table{width:100%}th{text-align:left;color:#888;font-size:11px;padding:10px;border-bottom:1px solid #333}td{padding:10px}.direction{padding:3px 8px;border-radius:4px;font-size:11px}.BUY{background:#00cc96;color:#000}.SELL{background:#ef553b;color:#fff}.modules{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}.module{background:#222;padding:10px;border-radius:8px;text-align:center}.progress{height:10px;background:#222;border-radius:5px;margin:15px 0}.fill{height:100%;background:linear-gradient(90deg,#00cc96,#00d4ff);border-radius:5px}</style></head><body><div class=header><div class=logo>Eko Trading System</div><div class=status> ONLINE</div></div>METRICS<div class=section><h3>Daily Progress (+5% target)</h3><div class=progress><div class=fill style=width:PROGRESS%></div></div></div><div class=section><h3>System Modules</h3><div class=modules>MODULES</div></div><div class=section><h3>Recent Trades</h3><table><tr><th>Symbol</th><th>Dir</th><th>P&L</th><th>Status</th><th>Time</th></tr>TRADES</table></div><p style=text-align:center;color:#666;font-size:12px>TIMESTAMP</p></body></html>'''

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        state = json.loads(Path('unified_brain_state.json').read_text()) if Path('unified_brain_state.json').exists() else {}
        db = json.loads(Path('db/unified_trading.db').read_text()) if Path('db/unified_trading.db').exists() else {}
        trades = db.get('trades', [])
        stats = state.get('stats', {})
        modules = state.get('modules', {})
        daily = stats.get('daily_pnl_pct', 0)
        progress = min(abs(daily) / 5 * 100, 100) if daily != 0 else 0
        metrics = f'<div class=metrics><div class=metric><div class=value>{len(trades)}</div><div class=label>Total Trades</div></div><div class=metric><div class=\"value positive\">{daily:+,.2f}%</div><div class=label>Daily P&L</div></div><div class=metric><div class=value>{stats.get(\"cycles\", 0)}</div><div class=label>Cycles</div></div><div class=metric><div class=value>{stats.get(\"trades_today\", 0)}</div><div class=label>Trades Today</div></div></div>'
        modules_html = ''.join(f'<div class=module>OK {k.replace(\"_\", \" \").title()}</div>' for k,v in modules.items())
        trades_html = ''.join(f'<tr><td>{t.get(\"symbol\", \"\")}</td><td><span class=\"direction {t.get(\"direction\", \"\")}\">{t.get(\"direction\", \"\")}</span></td><td class=\"positive\">{t.get(\"pnl_pct\") or 0:+,.2f}%</td><td>{t.get(\"status\", \"\")}</td><td style=\"color:#666\">{str(t.get(\"timestamp\", \"\"))[:19]}</td></tr>' for t in reversed(trades[-15:]))
        html = HTML.replace('METRICS', metrics).replace('PROGRESS', f'{progress:.1f}').replace('MODULES', modules_html).replace('TRADES', trades_html).replace('TIMESTAMP', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    def log_message(self, *args): pass

HTTPServer(('', 8502), H).serve_forever()
"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "$(date): Dashboard crashed with exit code $EXIT_CODE"
    else
        echo "$(date): Dashboard stopped"
    fi
    sleep 2
done
