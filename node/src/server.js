/**
 * Dual Apex Core System - REST API Server
 * Provides HTTP endpoints for system monitoring and control
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const fs = require('fs').promises;
const path = require('path');
const WebSocket = require('ws');

const app = express();
const PORT = process.env.API_PORT || 8889;
const WS_PORT = process.env.WS_PORT || 8890;

// Middleware
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json());
app.use(morgan('combined'));

// In-memory storage for demo (replace with Redis/PostgreSQL in production)
const systemMetrics = {
    totalProfit: 0,
    tradeCount: 0,
    winRate: 0,
    startTime: Date.now(),
    status: 'running'
};

const recentTrades = [];
const activeSignals = [];

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
    const uptime = (Date.now() - systemMetrics.startTime) / 1000;
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: `${(uptime / 3600).toFixed(1)} hours`,
        version: '1.0.0'
    });
});

/**
 * Get current trading signals
 */
app.get('/signals', async (req, res) => {
    try {
        const { strategy, confidence } = req.query;
        
        let filtered = [...activeSignals];
        
        if (strategy) {
            filtered = filtered.filter(s => s.strategy === strategy);
        }
        
        if (confidence) {
            filtered = filtered.filter(s => s.confidence >= parseFloat(confidence));
        }
        
        res.json({
            signals: filtered,
            count: filtered.length,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Error getting signals:', error);
        res.status(500).json({ error: 'Failed to fetch signals' });
    }
});

/**
 * Get recent trades
 */
app.get('/trades', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 10;
        const strategy = req.query.strategy;
        
        let filtered = [...recentTrades];
        
        if (strategy) {
            filtered = filtered.filter(t => t.strategy === strategy);
        }
        
        const result = filtered.slice(-limit).reverse();
        
        res.json({
            trades: result,
            count: result.length,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Error getting trades:', error);
        res.status(500).json({ error: 'Failed to fetch trades' });
    }
});

/**
 * Get performance metrics
 */
app.get('/metrics', async (req, res) => {
    try {
        // Try to read metrics from Python orchestrator
        let rustMetrics = {};
        try {
            const metricsPath = '/tmp/apex_metrics.json';
            const data = await fs.readFile(metricsPath, 'utf8');
            const parsed = JSON.parse(data);
            rustMetrics = parsed.rust_metrics || {};
        } catch (e) {
            // Metrics file not available yet
        }
        
        const uptime = (Date.now() - systemMetrics.startTime) / 1000;
        
        // Calculate strategy breakdown
        const strategyBreakdown = {};
        recentTrades.forEach(trade => {
            if (!strategyBreakdown[trade.strategy]) {
                strategyBreakdown[trade.strategy] = {
                    profit: 0,
                    trades: 0,
                    wins: 0
                };
            }
            strategyBreakdown[trade.strategy].profit += trade.profit;
            strategyBreakdown[trade.strategy].trades += 1;
            if (trade.success) {
                strategyBreakdown[trade.strategy].wins += 1;
            }
        });
        
        res.json({
            total_trades: recentTrades.length,
            total_profit: rustMetrics.total_profit || systemMetrics.totalProfit,
            win_rate: rustMetrics.win_rate || systemMetrics.winRate,
            avg_profit: rustMetrics.avg_profit || 0,
            avg_execution_time_ms: rustMetrics.avg_execution_time_ms || 0,
            uptime_hours: (uptime / 3600).toFixed(2),
            strategy_breakdown: strategyBreakdown,
            rust_engine_active: Object.keys(rustMetrics).length > 0,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Error getting metrics:', error);
        res.status(500).json({ error: 'Failed to fetch metrics' });
    }
});

/**
 * Execute a trade (POST)
 */
app.post('/execute', async (req, res) => {
    try {
        const { strategy, position_id, amount } = req.body;
        
        if (!strategy || !position_id) {
            return res.status(400).json({ 
                error: 'Missing required fields: strategy, position_id' 
            });
        }
        
        // Generate trade ID
        const tradeId = `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Queue for execution
        console.log(`Queuing trade: ${strategy} - ${position_id}`);
        
        res.json({
            status: 'queued',
            trade_id: tradeId,
            strategy,
            position_id,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Error executing trade:', error);
        res.status(500).json({ error: 'Failed to queue trade' });
    }
});

/**
 * Get system status
 */
app.get('/status', (req, res) => {
    res.json({
        status: systemMetrics.status,
        uptime: (Date.now() - systemMetrics.startTime) / 1000,
        active_strategies: ['liquidation_hunting', 'cross_chain_arbitrage', 'pump_prediction'],
        rust_engine: true,
        ml_inference: true,
        timestamp: new Date().toISOString()
    });
});

/**
 * Get dashboard data (aggregated)
 */
app.get('/dashboard', async (req, res) => {
    try {
        const uptime = (Date.now() - systemMetrics.startTime) / 1000;
        
        // Last 24h profit
        const last24h = recentTrades.filter(t => {
            return (Date.now() - new Date(t.timestamp).getTime()) < 24 * 3600 * 1000;
        });
        
        const profit24h = last24h.reduce((sum, t) => sum + t.profit, 0);
        
        res.json({
            overview: {
                total_profit: systemMetrics.totalProfit,
                profit_24h: profit24h,
                trade_count: recentTrades.length,
                win_rate: systemMetrics.winRate,
                uptime_hours: (uptime / 3600).toFixed(1)
            },
            recent_signals: activeSignals.slice(-5),
            recent_trades: recentTrades.slice(-10).reverse(),
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Error getting dashboard:', error);
        res.status(500).json({ error: 'Failed to fetch dashboard data' });
    }
});

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ port: WS_PORT });

wss.on('connection', (ws) => {
    console.log('WebSocket client connected');
    
    ws.send(JSON.stringify({
        type: 'connection',
        message: 'Connected to Dual Apex Core System',
        timestamp: new Date().toISOString()
    }));
    
    ws.on('message', (message) => {
        console.log('Received:', message.toString());
    });
    
    ws.on('close', () => {
        console.log('WebSocket client disconnected');
    });
});

// Broadcast updates to all WebSocket clients
function broadcastUpdate(type, data) {
    const message = JSON.stringify({
        type,
        data,
        timestamp: new Date().toISOString()
    });
    
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// Simulate periodic updates (replace with real data stream)
setInterval(() => {
    // Add mock signal
    if (Math.random() > 0.7) {
        const signal = {
            strategy: ['liquidation_hunting', 'cross_chain_arbitrage', 'pump_prediction'][Math.floor(Math.random() * 3)],
            pair: ['WETH', 'WBTC', 'USDC', 'MATIC'][Math.floor(Math.random() * 4)],
            confidence: 0.75 + Math.random() * 0.2,
            expected_profit: 1000 + Math.random() * 50000,
            timestamp: new Date().toISOString()
        };
        
        activeSignals.push(signal);
        if (activeSignals.length > 20) activeSignals.shift();
        
        broadcastUpdate('signal', signal);
    }
    
    // Add mock trade
    if (Math.random() > 0.8) {
        const trade = {
            strategy: ['liquidation_hunting', 'cross_chain_arbitrage', 'pump_prediction'][Math.floor(Math.random() * 3)],
            pair: ['WETH', 'WBTC', 'USDC'][Math.floor(Math.random() * 3)],
            success: Math.random() > 0.2,
            profit: (Math.random() - 0.2) * 10000,
            tx_hash: `0x${Math.random().toString(16).substr(2, 64)}`,
            gas_used: 150000 + Math.floor(Math.random() * 50000),
            execution_time_ms: 50 + Math.floor(Math.random() * 200),
            timestamp: new Date().toISOString()
        };
        
        recentTrades.push(trade);
        if (recentTrades.length > 100) recentTrades.shift();
        
        systemMetrics.totalProfit += trade.profit;
        systemMetrics.tradeCount += 1;
        
        const wins = recentTrades.filter(t => t.success).length;
        systemMetrics.winRate = wins / recentTrades.length;
        
        broadcastUpdate('trade', trade);
    }
}, 5000);

// Start HTTP server
app.listen(PORT, () => {
    console.log(`🚀 Dual Apex API Server running on port ${PORT}`);
    console.log(`📡 WebSocket server running on port ${WS_PORT}`);
    console.log(`📊 Dashboard: http://localhost:${PORT}/dashboard`);
    console.log(`🏥 Health: http://localhost:${PORT}/health`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Shutting down gracefully...');
    systemMetrics.status = 'shutting_down';
    process.exit(0);
});
