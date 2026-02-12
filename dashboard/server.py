#!/usr/bin/env python3
"""
Simple HTTP Dashboard Server
============================
Serves the HTML dashboard at http://localhost:8502

Usage:
    python3 dashboard_server.py
    python3 dashboard_server.py --port 8502
"""

import http.server
import socketserver
import json
from pathlib import Path
from datetime import datetime
import threading
import time

PORT = 8502
DIRECTORY = Path(__file__).parent


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for the dashboard."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/api/status":
            self.send_status()
        else:
            super().do_GET()
    
    def send_status(self):
        """Send status as JSON."""
        status = {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "wallet": {
                "address": "H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a",
                "balance_sol": 0.0,
                "balance_usd": 0.0,
                "network": "devnet"
            },
            "price": {
                "sol": 80.76,
                "source": "Jupiter DEX"
            },
            "agents": {
                "coordinator": "active",
                "researcher": "active",
                "trader": "idle",
                "risk_manager": "active",
                "ux_manager": "idle"
            },
            "trades": {
                "today": 0,
                "pending": 0,
                "completed": 0
            }
        }
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode())
    
    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def run_server():
    """Run the dashboard server."""
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"🌐 Dashboard Server")
        print(f"   URL: http://localhost:{PORT}")
        print(f"   Local: http://{get_local_ip()}:{PORT}")
        print(f"   File: {DIRECTORY / 'index.html'}")
        print("-" * 40)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")


def get_local_ip():
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard HTTP Server")
    parser.add_argument("--port", "-p", type=int, default=PORT,
                       help=f"Server port (default: {PORT})")
    
    args = parser.parse_args()
    
    run_server()
