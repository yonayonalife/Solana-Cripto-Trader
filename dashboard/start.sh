#!/bin/bash
#
# Dashboard Launcher
# ==================
# Starts the HTML dashboard on port 8502
#
# Usage:
#   ./start_dashboard.sh      # Start in background
#   ./start_dashboard.sh stop # Stop the server
#

PORT=8502
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="/tmp/dashboard_${PORT}.pid"

start() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Dashboard already running on port $PORT (PID: $OLD_PID)"
            return
        else
            rm "$PID_FILE"
        fi
    fi
    
    echo "🚀 Starting Dashboard on port $PORT..."
    cd "$DIR"
    nohup python3 -m http.server $PORT > /tmp/dashboard.log 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    sleep 2
    
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Dashboard running!"
        echo "   URL: http://localhost:$PORT"
        echo "   PID: $PID"
    else
        echo "❌ Failed to start dashboard"
        rm "$PID_FILE"
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo "🛑 Stopping dashboard (PID: $PID)..."
            kill $PID
            rm "$PID_FILE"
            echo "✅ Stopped"
        else
            echo "Dashboard not running (stale PID file)"
            rm "$PID_FILE"
        fi
    else
        echo "Dashboard not running"
    fi
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo "✅ Dashboard running on port $PORT (PID: $PID)"
            echo "   URL: http://localhost:$PORT"
        else
            echo "❌ Dashboard not running (stale PID file)"
        fi
    else
        echo "❌ Dashboard not running"
    fi
}

case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
