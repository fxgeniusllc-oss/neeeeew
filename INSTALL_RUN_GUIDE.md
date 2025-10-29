# 🚀 Installation, Run & Usage Guide - Dual Apex Core System

## Table of Contents
- [System Overview](#system-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Usage Guide](#usage-guide)
- [Monitoring & Debugging](#monitoring--debugging)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## System Overview

The Dual Apex Core System is a hybrid DeFi trading platform combining:
- **Rust**: Ultra-fast trading engine with ML inference (via PyO3)
- **Python**: Strategy orchestration, ML training, system coordination
- **Node.js**: REST API, WebSocket server, metrics aggregation
- **Frontend**: Real-time dashboard with live updates

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                    │
│                  Real-time Dashboard                     │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/WebSocket
┌─────────────────────▼───────────────────────────────────┐
│              Node.js API Server (Express)                │
│         REST API + WebSocket + Metrics Server            │
└─────────────────────┬───────────────────────────────────┘
                      │ IPC/File
┌─────────────────────▼───────────────────────────────────┐
│           Python Orchestrator (asyncio)                  │
│    Strategy Coordination + ML Training + Signals         │
└───────────┬─────────────────────────────────┬───────────┘
            │                                 │
    ┌───────▼────────┐              ┌────────▼────────┐
    │  Rust Engine   │              │  ML Models      │
    │  (PyO3 Bridge) │              │  (XGBoost)      │
    │  Ultra-fast    │              │  Training &     │
    │  Execution     │              │  Inference      │
    └────────────────┘              └─────────────────┘
            │
    ┌───────▼────────┐
    │  PostgreSQL    │
    │  Trade History │
    └────────────────┘
```

---

## Prerequisites

### Required Software

1. **Rust** (v1.75+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Python** (v3.9+)
   ```bash
   python3 --version  # Should be 3.9 or higher
   ```

3. **Node.js** (v16+)
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

4. **PostgreSQL** (v13+)
   ```bash
   sudo apt-get update
   sudo apt-get install -y postgresql postgresql-contrib
   ```

5. **Redis** (v6+)
   ```bash
   sudo apt-get install -y redis-server
   ```

### Optional but Recommended

- **Docker & Docker Compose** (for containerized deployment)
  ```bash
  sudo apt-get install -y docker.io docker-compose
  sudo usermod -aG docker $USER
  ```

- **Git** (for cloning repository)
  ```bash
  sudo apt-get install -y git
  ```

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB minimum
- **OS**: Ubuntu 20.04+, Debian 11+, or compatible Linux
- **Network**: Stable internet connection (for blockchain RPC calls)

---

## Installation

### Method 1: Local Development Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/fxgeniusllc-oss/neeeeew.git
cd neeeeew
```

#### Step 2: Build Rust Engine

```bash
# Install Rust dependencies
cd rust
cargo build --release

# Build Python bindings (PyO3)
cd ..
pip install maturin
maturin develop --release

# Verify Rust module is available
python3 -c "import dual_apex_engine; print('Rust engine loaded successfully')"
```

#### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy
```

#### Step 4: Install Node.js Dependencies

```bash
npm install
```

#### Step 5: Initialize Database

```bash
# Start PostgreSQL (if not running)
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE dual_apex;
CREATE USER apex_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE dual_apex TO apex_user;
EOF

# Initialize database schema
python3 python/scripts/init_database.py
```

#### Step 6: Start Redis

```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Step 7: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (use your favorite editor)
nano .env
```

**Important**: Update these values in `.env`:
- `POLYGON_RPC`: Your Alchemy/Infura Polygon RPC URL
- `ETHEREUM_RPC`: Your Ethereum RPC URL
- `PRIVATE_KEY`: Your wallet private key (NEVER commit this!)
- `POSTGRES_PASSWORD`: Your PostgreSQL password
- `TELEGRAM_TOKEN`: Your Telegram bot token (optional)

---

### Method 2: Docker Deployment

#### Quick Start with Docker Compose

```bash
# Clone repository
git clone https://github.com/fxgeniusllc-oss/neeeeew.git
cd neeeeew

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

This starts:
- PostgreSQL database (port 5432)
- Redis cache (port 6379)
- Main application (ports 8888, 8889, 8890)
- Nginx reverse proxy (port 80)

---

## Configuration

### Environment Variables

Edit `.env` file with your settings:

```bash
# Blockchain RPC Endpoints
POLYGON_RPC=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
ETHEREUM_RPC=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
ARBITRUM_RPC=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY

# Private Key (CRITICAL: Keep secure!)
PRIVATE_KEY=0xYOUR_PRIVATE_KEY

# Database
POSTGRES_HOST=localhost
POSTGRES_DB=dual_apex
POSTGRES_USER=apex_user
POSTGRES_PASSWORD=your_secure_password

# Capital & Risk
BASE_CAPITAL=100000
MAX_POSITION_SIZE=1000000
MAX_DAILY_LOSS=100000
MAX_GAS_PRICE=200

# API Ports
API_PORT=8889
WS_PORT=8890
DASHBOARD_PORT=8888
```

### Strategy Configuration

Strategies can be configured in Python code or via config files. Default settings:

```python
# Liquidation Hunting
min_confidence = 0.80      # 80% ML confidence threshold
min_profit = 5000          # $5k minimum profit

# Cross-Chain Arbitrage
min_spread = 0.02          # 2% minimum spread
bridge_timeout = 1800      # 30 min max bridge time

# Pump Prediction
min_confidence = 0.85      # 85% confidence threshold
stop_loss = 0.15           # 15% stop loss
```

---

## Running the System

### Option 1: Run All Components Separately

**Terminal 1 - Python Orchestrator:**
```bash
source venv/bin/activate
python3 python/orchestrator.py
```

**Terminal 2 - Node.js API:**
```bash
node node/src/server.js
```

**Terminal 3 - Frontend Dashboard:**
```bash
# Serve dashboard (simple HTTP server)
cd frontend
python3 -m http.server 8888

# Or use Node.js
npx http-server -p 8888
```

### Option 2: Run with Process Manager (Recommended)

Using **tmux** or **screen**:

```bash
# Start new tmux session
tmux new -s apex

# Window 1: Python
python3 python/orchestrator.py

# Ctrl+B then C to create new window
# Window 2: Node.js
node node/src/server.js

# Ctrl+B then C
# Window 3: Dashboard
cd frontend && python3 -m http.server 8888

# Detach: Ctrl+B then D
# Reattach: tmux attach -t apex
```

### Option 3: Docker Compose (Easiest)

```bash
docker-compose up -d
```

---

## Usage Guide

### Accessing the Dashboard

Open your browser:
```
http://localhost:8888
```

or (with Docker + Nginx):
```
http://localhost/
```

**Dashboard Features:**
- Real-time profit/loss tracking
- Active trading signals
- Recent trade history
- Performance metrics
- Win rate and Sharpe ratio

### REST API Endpoints

Base URL: `http://localhost:8889`

#### 1. Health Check
```bash
curl http://localhost:8889/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": "2.5 hours"
}
```

#### 2. Get Trading Signals
```bash
curl http://localhost:8889/signals?strategy=liquidation_hunting&confidence=0.8
```

Response:
```json
{
  "signals": [
    {
      "strategy": "liquidation_hunting",
      "pair": "WETH",
      "confidence": 0.94,
      "expected_profit": 45200,
      "timestamp": "2024-01-15T10:30:15Z"
    }
  ],
  "count": 1
}
```

#### 3. Get Recent Trades
```bash
curl http://localhost:8889/trades?limit=10
```

#### 4. Get Metrics
```bash
curl http://localhost:8889/metrics
```

Response:
```json
{
  "total_trades": 847,
  "total_profit": 547230,
  "win_rate": 0.873,
  "avg_execution_time_ms": 125,
  "strategy_breakdown": {
    "liquidation_hunting": {
      "profit": 245600,
      "trades": 234
    }
  }
}
```

#### 5. Execute Trade (Manual)
```bash
curl -X POST http://localhost:8889/execute \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "liquidation_hunting",
    "position_id": "0x7a8b9c...",
    "amount": 50000
  }'
```

### WebSocket Connection

Connect to real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8890');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

Message types:
- `signal`: New trading signal detected
- `trade`: Trade executed
- `connection`: Initial connection

### Python API Usage

```python
# Import Rust engine
from dual_apex_engine import TradingEngine, TradeSignal

# Create engine
engine = TradingEngine()

# Add signal
signal = TradeSignal(
    strategy="liquidation_hunting",
    pair="WETH",
    direction="liquidate",
    confidence=0.92,
    expected_profit=15000
)
engine.add_signal(signal)

# Execute signals
results = engine.execute_signals(max_gas_price=200)

# Get metrics
metrics = engine.get_metrics()
print(f"Total profit: ${metrics['total_profit']:.2f}")
print(f"Win rate: {metrics['win_rate']:.2%}")
```

### Training ML Models

```bash
# Train liquidation prediction model
cd python/ml
python3 trainer.py

# Models saved to: models/liquidation_model.json
```

From Python:
```python
from ml.trainer import MLModelTrainer

trainer = MLModelTrainer({})
results = trainer.train_liquidation_model()
print(f"Model accuracy: {results['accuracy']:.3f}")
```

---

## Monitoring & Debugging

### Logs

**Python Orchestrator:**
```bash
tail -f logs/orchestrator.log
```

**Node.js API:**
```bash
tail -f logs/api.log
```

**Docker:**
```bash
docker-compose logs -f app
```

### Database Queries

Check trade history:
```sql
-- Connect to database
psql -U apex_user -d dual_apex

-- Recent trades
SELECT strategy, pair, profit, status, timestamp 
FROM trades 
ORDER BY timestamp DESC 
LIMIT 10;

-- Strategy performance
SELECT 
    strategy, 
    COUNT(*) as trades,
    SUM(profit) as total_profit,
    AVG(profit) as avg_profit,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)::float / COUNT(*) as win_rate
FROM trades 
GROUP BY strategy;
```

### Redis Cache

```bash
redis-cli

# Check active signals
KEYS signal:*

# View metrics
GET metrics:total_profit
```

### Performance Metrics

Monitor system resources:
```bash
# CPU & Memory
htop

# Docker resources
docker stats

# Network usage
iftop
```

---

## Production Deployment

### Security Hardening

1. **Firewall Configuration**
```bash
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable
```

2. **SSL/TLS Setup**
```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com

# Update nginx config with SSL
```

3. **Secure Private Keys**
```bash
# Use hardware wallet or secure key management
# Never commit .env to git
# Rotate API keys regularly
```

4. **Database Security**
```bash
# Use strong passwords
# Enable SSL connections
# Restrict network access
```

### Systemd Service

Create `/etc/systemd/system/dual-apex.service`:

```ini
[Unit]
Description=Dual Apex Core System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=apex
WorkingDirectory=/opt/dual-apex
Environment="PATH=/opt/dual-apex/venv/bin"
ExecStart=/opt/dual-apex/venv/bin/python3 python/orchestrator.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable dual-apex
sudo systemctl start dual-apex
sudo systemctl status dual-apex
```

### Backup Strategy

```bash
# Backup database daily
0 2 * * * pg_dump dual_apex > /backup/apex_$(date +\%Y\%m\%d).sql

# Backup configuration
0 3 * * * tar -czf /backup/config_$(date +\%Y\%m\%d).tar.gz .env config/

# Backup trade logs
0 4 * * * tar -czf /backup/logs_$(date +\%Y\%m\%d).tar.gz logs/
```

### Monitoring Setup

Use Prometheus + Grafana for advanced monitoring:

1. **Install Prometheus**
2. **Configure Node Exporter**
3. **Set up Grafana dashboards**
4. **Configure alerts** (email, Slack, Telegram)

---

## Troubleshooting

### Issue: Rust module not found

```bash
# Rebuild Rust engine
cd rust
cargo clean
cargo build --release

# Reinstall Python bindings
pip install maturin
maturin develop --release
```

### Issue: Database connection failed

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U apex_user -d dual_apex -c "SELECT 1"

# Reset database
python3 python/scripts/init_database.py
```

### Issue: Node.js server won't start

```bash
# Check port availability
netstat -tulpn | grep 8889

# Kill existing process
pkill -f "node.*server.js"

# Reinstall dependencies
rm -rf node_modules
npm install
```

### Issue: Dashboard not loading

```bash
# Check API is running
curl http://localhost:8889/health

# Check WebSocket
wscat -c ws://localhost:8890

# Browser console for errors
# Open DevTools (F12) and check Console tab
```

### Issue: High memory usage

```bash
# Check processes
docker stats  # for Docker
ps aux | grep python

# Adjust Python memory limits
# Add to orchestrator.py:
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
```

### Issue: Slow execution

```bash
# Check Rust engine is being used
grep "Rust engine" logs/orchestrator.log

# Verify ML inference
python3 -c "from dual_apex_engine import MLInferenceEngine; print('OK')"

# Profile Python code
python3 -m cProfile python/orchestrator.py
```

---

## Additional Resources

- **Documentation**: See `README.md` for detailed architecture
- **API Reference**: OpenAPI spec at `/api/docs`
- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Community support (link in README)

---

## Quick Reference Commands

```bash
# Start system (development)
python3 python/orchestrator.py & node node/src/server.js

# Start system (Docker)
docker-compose up -d

# View logs
tail -f logs/orchestrator.log
docker-compose logs -f

# Check status
curl http://localhost:8889/health
curl http://localhost:8889/metrics

# Stop system
pkill -f "python.*orchestrator"
pkill -f "node.*server"
docker-compose down

# Database backup
pg_dump dual_apex > backup.sql

# Restore database
psql dual_apex < backup.sql
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/fxgeniusllc-oss/neeeeew/issues
- Email: support@fxgeniusllc.com

---

**Built with ❤️ for institutional-grade DeFi trading**
