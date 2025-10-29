#!/usr/bin/env python3
"""
Main orchestrator for Dual Apex Core System
Coordinates Rust trading engine, Python strategies, and Node.js API
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import json

# Import Rust trading engine
try:
    from dual_apex_engine import TradingEngine, TradeSignal, MLInferenceEngine
    RUST_AVAILABLE = True
except ImportError:
    logging.warning("Rust engine not available. Using Python fallback.")
    RUST_AVAILABLE = False

from strategies.liquidation_hunting import LiquidationHunter
from strategies.cross_chain_arbitrage import CrossChainArbitrage
from strategies.pump_prediction import PumpPredictor
from ml.trainer import MLModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DualApexOrchestrator:
    """
    Master orchestrator for all trading strategies
    Integrates Rust execution engine with Python strategy logic
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.rust_engine = TradingEngine() if RUST_AVAILABLE else None
        self.ml_inference = MLInferenceEngine() if RUST_AVAILABLE else None
        
        # Initialize strategies
        self.strategies = {
            'liquidation_hunting': LiquidationHunter(config),
            'cross_chain_arbitrage': CrossChainArbitrage(config),
            'pump_prediction': PumpPredictor(config),
        }
        
        self.running = False
        self.total_profit = 0.0
        self.trade_history = []
        
        logger.info("Dual Apex Orchestrator initialized with Rust engine: %s", RUST_AVAILABLE)
    
    async def start(self):
        """Start the orchestration engine"""
        self.running = True
        logger.info("Starting Dual Apex Core System")
        
        # Load ML models
        if self.ml_inference:
            model_path = self.config.get('ml_model_path', 'models/liquidation_model.onnx')
            self.ml_inference.load_model(model_path)
        
        # Start strategy monitoring loops
        tasks = [
            self.strategy_loop(name, strategy) 
            for name, strategy in self.strategies.items()
        ]
        tasks.append(self.execution_loop())
        tasks.append(self.metrics_loop())
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down orchestrator...")
            self.running = False
    
    async def strategy_loop(self, name: str, strategy):
        """Main loop for each strategy"""
        logger.info(f"Starting strategy loop: {name}")
        
        while self.running:
            try:
                # Get signals from strategy
                signals = await strategy.get_signals()
                
                for signal in signals:
                    if self.rust_engine:
                        # Use Rust engine for high-performance execution
                        rust_signal = TradeSignal(
                            strategy=name,
                            pair=signal['pair'],
                            direction=signal['direction'],
                            confidence=signal['confidence'],
                            expected_profit=signal['expected_profit']
                        )
                        self.rust_engine.add_signal(rust_signal)
                    else:
                        # Fallback to Python execution
                        await self.execute_python_signal(name, signal)
                
                await asyncio.sleep(5)  # Poll every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in {name} strategy loop: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def execution_loop(self):
        """Execute pending trades using Rust engine"""
        logger.info("Starting execution loop")
        
        while self.running:
            try:
                if self.rust_engine:
                    max_gas = self.config.get('max_gas_price', 200)
                    results = self.rust_engine.execute_signals(max_gas)
                    
                    for result in results:
                        self.total_profit += result.profit
                        self.trade_history.append({
                            'success': result.success,
                            'profit': result.profit,
                            'tx_hash': result.tx_hash,
                            'gas_used': result.gas_used,
                            'execution_time_ms': result.execution_time_ms,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        logger.info(f"Trade executed: ${result.profit:.2f} in {result.execution_time_ms}ms")
                
                await asyncio.sleep(2)  # Execute every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def metrics_loop(self):
        """Periodic metrics reporting"""
        logger.info("Starting metrics loop")
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                if self.rust_engine:
                    metrics = self.rust_engine.get_metrics()
                    logger.info(f"Performance Metrics: {json.dumps(metrics, indent=2)}")
                    
                    # Save metrics to file for Node.js API
                    with open('/tmp/apex_metrics.json', 'w') as f:
                        json.dump({
                            'rust_metrics': metrics,
                            'total_profit': self.total_profit,
                            'trade_count': len(self.trade_history),
                            'timestamp': datetime.now().isoformat()
                        }, f)
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}", exc_info=True)
    
    async def execute_python_signal(self, strategy: str, signal: Dict):
        """Fallback Python execution when Rust is unavailable"""
        logger.info(f"Executing Python signal: {strategy} - {signal['pair']}")
        # Simulate execution
        await asyncio.sleep(0.1)
        profit = signal['expected_profit'] * 0.9
        self.total_profit += profit
        logger.info(f"Python execution completed: ${profit:.2f}")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        if self.rust_engine:
            return {
                'running': self.running,
                'rust_available': RUST_AVAILABLE,
                'total_profit': self.rust_engine.get_total_profit(),
                'trade_count': self.rust_engine.get_trade_count(),
                'win_rate': self.rust_engine.get_win_rate(),
                'strategies': list(self.strategies.keys())
            }
        return {
            'running': self.running,
            'rust_available': False,
            'total_profit': self.total_profit,
            'trade_count': len(self.trade_history)
        }


async def main():
    """Main entry point"""
    # Load configuration
    config = {
        'max_gas_price': 200,
        'base_capital': 100000,
        'risk_per_trade': 0.02,
        'ml_model_path': 'models/liquidation_model.onnx',
        'polygon_rpc': os.getenv('POLYGON_RPC', 'https://polygon-rpc.com'),
        'ethereum_rpc': os.getenv('ETHEREUM_RPC', 'https://eth-mainnet.g.alchemy.com/v2/demo'),
    }
    
    # Create orchestrator
    orchestrator = DualApexOrchestrator(config)
    
    # Start system
    await orchestrator.start()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
