#!/bin/bash
# Dual Apex Core System - Startup Script
# Starts all components in the correct order

set -e

echo "🚀 Starting Dual Apex Core System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${GREEN}✓${NC} Running in Docker container"
    IS_DOCKER=true
else
    IS_DOCKER=false
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."

if ! $IS_DOCKER; then
    if command_exists node; then
        echo -e "${GREEN}✓${NC} Node.js $(node --version)"
    else
        echo -e "${RED}✗${NC} Node.js not found. Please install Node.js 16+"
        exit 1
    fi

    if command_exists python3; then
        echo -e "${GREEN}✓${NC} Python $(python3 --version)"
    else
        echo -e "${RED}✗${NC} Python3 not found. Please install Python 3.9+"
        exit 1
    fi

    if command_exists cargo; then
        echo -e "${GREEN}✓${NC} Rust $(rustc --version)"
    else
        echo -e "${YELLOW}⚠${NC} Rust not found. Rust engine will be disabled."
    fi
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠${NC} .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}⚠${NC} Please edit .env with your settings before running."
    exit 1
fi

# Create necessary directories
mkdir -p logs data models

# Check database
if [ ! -f data/dual_apex.db ]; then
    echo ""
    echo "Initializing database..."
    python3 python/scripts/init_database.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Database initialized"
    else
        echo -e "${RED}✗${NC} Database initialization failed"
        exit 1
    fi
fi

# Start components
echo ""
echo "Starting services..."

# Start Node.js API server
echo "Starting Node.js API server (port 8889)..."
node node/src/server.js > logs/node_api.log 2>&1 &
NODE_PID=$!
echo -e "${GREEN}✓${NC} Node.js API started (PID: $NODE_PID)"

# Wait for API to be ready
sleep 3

# Check if API is responding
if curl -s http://localhost:8889/health > /dev/null; then
    echo -e "${GREEN}✓${NC} API is responding"
else
    echo -e "${RED}✗${NC} API is not responding"
    kill $NODE_PID 2>/dev/null || true
    exit 1
fi

# Start Python orchestrator (if not in Docker)
if ! $IS_DOCKER; then
    echo "Starting Python orchestrator..."
    python3 python/orchestrator.py > logs/orchestrator.log 2>&1 &
    PYTHON_PID=$!
    echo -e "${GREEN}✓${NC} Python orchestrator started (PID: $PYTHON_PID)"
fi

# Start frontend server
echo "Starting frontend dashboard (port 8888)..."
cd frontend
python3 -m http.server 8888 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}✓${NC} Frontend server started (PID: $FRONTEND_PID)"

# Save PIDs for cleanup
echo "$NODE_PID" > logs/node.pid
if ! $IS_DOCKER; then
    echo "$PYTHON_PID" > logs/python.pid
fi
echo "$FRONTEND_PID" > logs/frontend.pid

# Success message
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Dual Apex Core System Started Successfully! 🚀${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Access points:"
echo "  📊 Dashboard:  http://localhost:8888/dashboard.html"
echo "  🔗 REST API:   http://localhost:8889"
echo "  📡 WebSocket:  ws://localhost:8890"
echo "  🏥 Health:     http://localhost:8889/health"
echo ""
echo "Logs:"
echo "  Node.js API:        tail -f logs/node_api.log"
if ! $IS_DOCKER; then
    echo "  Python Orchestrator: tail -f logs/orchestrator.log"
fi
echo "  Frontend:           tail -f logs/frontend.log"
echo ""
echo "To stop all services: ./stop.sh"
echo ""

# Keep script running (optional)
if [ "$1" == "--foreground" ] || [ "$1" == "-f" ]; then
    echo "Running in foreground mode. Press Ctrl+C to stop."
    echo ""
    tail -f logs/node_api.log &
    if ! $IS_DOCKER; then
        tail -f logs/orchestrator.log &
    fi
    wait
fi
