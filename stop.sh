#!/bin/bash
# Dual Apex Core System - Stop Script
# Stops all running components

echo "🛑 Stopping Dual Apex Core System..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Function to stop process by PID file
stop_process() {
    local name=$1
    local pidfile=$2
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null
            sleep 1
            if ps -p $pid > /dev/null 2>&1; then
                kill -9 $pid 2>/dev/null
            fi
            echo -e "${GREEN}✓${NC} Stopped $name (PID: $pid)"
        else
            echo -e "${RED}✗${NC} $name process not found (PID: $pid)"
        fi
        rm -f "$pidfile"
    else
        echo -e "${RED}✗${NC} No PID file for $name"
    fi
}

# Stop services
stop_process "Node.js API" "logs/node.pid"
stop_process "Python Orchestrator" "logs/python.pid"
stop_process "Frontend Server" "logs/frontend.pid"

# Also kill by process name (fallback)
pkill -f "node.*server.js" 2>/dev/null && echo -e "${GREEN}✓${NC} Killed remaining Node.js processes"
pkill -f "python.*orchestrator.py" 2>/dev/null && echo -e "${GREEN}✓${NC} Killed remaining Python processes"
pkill -f "python3 -m http.server 8888" 2>/dev/null && echo -e "${GREEN}✓${NC} Killed remaining frontend processes"

echo ""
echo -e "${GREEN}System stopped successfully.${NC}"
