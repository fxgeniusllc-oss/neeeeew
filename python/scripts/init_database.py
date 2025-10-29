#!/usr/bin/env python3
"""
Database Initialization Script
Creates PostgreSQL tables for trade history and metrics
"""

import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available. Using SQLite fallback.")
    POSTGRES_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logger.warning("sqlite3 not available")


def init_postgres_db():
    """Initialize PostgreSQL database"""
    # Get connection params from environment
    conn_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'dual_apex'),
        'user': os.getenv('POSTGRES_USER', 'apex_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'changeme')
    }
    
    logger.info(f"Connecting to PostgreSQL at {conn_params['host']}:{conn_params['port']}")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                strategy VARCHAR(50) NOT NULL,
                pair VARCHAR(20) NOT NULL,
                direction VARCHAR(20) NOT NULL,
                profit DECIMAL(18, 2) NOT NULL,
                confidence DECIMAL(5, 4),
                tx_hash VARCHAR(66),
                gas_used INTEGER,
                execution_time_ms INTEGER,
                status VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(50) NOT NULL,
                metric_value DECIMAL(18, 6) NOT NULL,
                strategy VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                strategy VARCHAR(50) NOT NULL,
                pair VARCHAR(20) NOT NULL,
                direction VARCHAR(20) NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                expected_profit DECIMAL(18, 2),
                status VARCHAR(20) DEFAULT 'pending',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
        """)
        
        conn.commit()
        logger.info("PostgreSQL database initialized successfully")
        
        # Insert sample data
        cursor.execute("""
            INSERT INTO metrics (metric_name, metric_value, strategy)
            VALUES ('initial_capital', 100000, 'system')
        """)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing PostgreSQL: {e}")
        return False


def init_sqlite_db():
    """Initialize SQLite database as fallback"""
    if not SQLITE_AVAILABLE:
        logger.error("SQLite not available")
        return False
    
    db_path = 'data/dual_apex.db'
    os.makedirs('data', exist_ok=True)
    
    logger.info(f"Initializing SQLite database at {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                profit REAL NOT NULL,
                confidence REAL,
                tx_hash TEXT,
                gas_used INTEGER,
                execution_time_ms INTEGER,
                status TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                strategy TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                expected_profit REAL,
                status TEXT DEFAULT 'pending',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("SQLite database initialized successfully")
        
        # Insert sample data
        cursor.execute("""
            INSERT INTO metrics (metric_name, metric_value, strategy)
            VALUES ('initial_capital', 100000, 'system')
        """)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing SQLite: {e}")
        return False


def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    if POSTGRES_AVAILABLE:
        success = init_postgres_db()
        if not success:
            logger.warning("PostgreSQL initialization failed, falling back to SQLite")
            success = init_sqlite_db()
    else:
        success = init_sqlite_db()
    
    if success:
        logger.info("Database initialization completed successfully")
        return 0
    else:
        logger.error("Database initialization failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
