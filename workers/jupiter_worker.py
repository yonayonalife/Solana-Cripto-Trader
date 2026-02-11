#!/usr/bin/env python3
"""
Jupiter Worker for Solana Trading Bot
===================================
Worker process that executes strategy mining tasks from the coordinator.

Based on: crypto_worker.py from Coinbase Cripto Trader

Features:
- Polls coordinator for work units
- Executes strategy mining with backtesting
- Submits results to coordinator
- Supports multi-instance execution
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.jupiter_client import JupiterClient, SOL_MINT, USDC_MINT
from tools.solana_wallet import SolanaWallet
from config.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jupiter_worker")


@dataclass
class WorkerConfig:
    """Worker configuration"""
    worker_id: str
    coordinator_url: str
    instance_number: int
    num_workers: int
    use_ray: bool = False


class JupiterWorker:
    """
    Worker that executes trading strategy mining tasks.
    
    Workflow:
    1. Poll coordinator for available work units
    2. Execute backtests on received work
    3. Submit results to coordinator
    4. Repeat
    """
    
    def __init__(
        self,
        config: WorkerConfig,
        wallet: Optional[SolanaWallet] = None,
        jupiter_client: Optional[JupiterClient] = None
    ):
        self.config = config
        self.wallet = wallet
        self.jupiter = jupiter_client or JupiterClient()
        
        # State
        self.running = True
        self.current_work_unit: Optional[Dict] = None
        self.start_time = datetime.now()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info(f"Worker {self.config.worker_id} shutting down...")
        self.running = False
    
    def get_worker_id(self) -> str:
        """Generate unique worker ID"""
        hostname = os.environ.get("HOSTNAME", "unknown")
        return f"{hostname}_W{self.config.instance_number}"
    
    def poll_for_work(self) -> Optional[Dict]:
        """
        Poll coordinator for available work units.
        
        Returns:
            Work unit dict or None if no work available
        """
        try:
            url = f"{self.config.coordinator_url}/api/get_work"
            params = {"worker_id": self.get_worker_id()}
            
            response = self.http_client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "success" and data.get("work_unit"):
                logger.info(f"Received work unit: {data['work_unit']['id']}")
                return data["work_unit"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error polling for work: {e}")
            return None
    
    def execute_work_unit(self, work_unit: Dict) -> Dict:
        """
        Execute a work unit and return results.
        
        Args:
            work_unit: Work unit with strategy parameters
        
        Returns:
            Results dict
        """
        logger.info(f"Executing work unit: {work_unit['id']}")
        
        strategy_params = work_unit.get("strategy_params", {})
        
        try:
            # Import here to avoid circular imports
            from backtesting.solana_backtester import (
                run_backtest,
                precompute_indicators,
                JupiterFees,
                generate_sample_data
            )
            
            # Generate or load data
            # In real implementation, this would load historical data
            df = generate_sample_data(n_candles=10000)
            
            # Pre-compute indicators
            indicators = precompute_indicators(df)
            
            # Run backtest (placeholder - would use actual genome)
            import numpy as np
            
            # Sample genome for testing
            genome = np.array([
                0.03,  # SL 3%
                0.06,  # TP 6%
                2,      # 2 rules
                0, 0, 4, 30, 0,   # RSI < 30
                0, 0, 4, 70, 1,   # RSI > 70
                0, 0, 0, 0, 0,
            ], dtype=np.float64)
            
            fees = JupiterFees()
            result = run_backtest(df, genome, initial_balance=1.0, fees=fees)
            
            # Format results
            results = {
                "work_unit_id": work_unit["id"],
                "worker_id": self.get_worker_id(),
                "pnl": result.pnl,
                "trades": result.trades,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "status": "completed",
                "completed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Work unit completed: PnL={result.pnl:.4f}, Trades={result.trades}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing work unit: {e}")
            return {
                "work_unit_id": work_unit["id"],
                "worker_id": self.get_worker_id(),
                "status": "error",
                "error": str(e)
            }
    
    def submit_results(self, results: Dict) -> bool:
        """
        Submit results to coordinator.
        
        Args:
            results: Results dict
        
        Returns:
            True if successful
        """
        try:
            url = f"{self.config.coordinator_url}/api/submit_result"
            
            response = self.http_client.post(
                url,
                json=results,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "success":
                logger.info(f"Results submitted successfully")
                return True
            else:
                logger.error(f"Failed to submit results: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting results: {e}")
            return False
    
    def register_with_coordinator(self) -> bool:
        """
        Register worker with coordinator.
        
        Returns:
            True if successful
        """
        try:
            import httpx
            
            url = f"{self.config.coordinator_url}/api/register_worker"
            data = {
                "worker_id": self.get_worker_id(),
                "hostname": os.environ.get("HOSTNAME", "unknown"),
                "platform": sys.platform,
                "instance_number": self.config.instance_number
            }
            
            response = httpx.post(url, json=data, timeout=10.0)
            response.raise_for_status()
            
            logger.info(f"Worker registered: {self.get_worker_id()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register: {e}")
            return False
    
    def heartbeat(self):
        """Send heartbeat to coordinator"""
        try:
            import httpx
            
            url = f"{self.config.coordinator_url}/api/heartbeat"
            data = {
                "worker_id": self.get_worker_id(),
                "status": "active",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
            
            httpx.post(url, json=data, timeout=5.0)
            
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
    
    def run(self):
        """
        Main worker loop.
        """
        import httpx
        
        self.http_client = httpx.Client(timeout=30.0)
        
        logger.info(f"Worker {self.get_worker_id()} starting...")
        
        # Register with coordinator
        if not self.register_with_coordinator():
            logger.warning("Failed to register, continuing anyway...")
        
        # Main loop
        poll_interval = 5.0  # seconds
        heartbeat_interval = 60.0  # seconds
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Poll for work
                work_unit = self.poll_for_work()
                
                if work_unit:
                    # Execute work
                    results = self.execute_work_unit(work_unit)
                    
                    # Submit results
                    self.submit_results(results)
                    
                    # Reset poll interval
                    poll_interval = 5.0
                else:
                    # No work available, wait longer
                    time.sleep(poll_interval)
                    poll_interval = min(poll_interval * 1.5, 60.0)
                
                # Send heartbeat
                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    self.heartbeat()
                    last_heartbeat = current_time
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10.0)
        
        # Cleanup
        self.http_client.close()
        logger.info(f"Worker {self.get_worker_id()} stopped")


def create_worker(config: WorkerConfig) -> JupiterWorker:
    """Create and configure worker"""
    return JupiterWorker(config)


def run_worker():
    """Entry point for worker"""
    import argparse
    import httpx
    
    parser = argparse.ArgumentParser(description="Jupiter Worker")
    parser.add_argument("--coordinator", type=str, help="Coordinator URL")
    parser.add_argument("--worker-id", type=str, help="Worker ID")
    parser.add_argument("--instance", type=int, default=1, help="Instance number")
    parser.add_argument("--num-workers", type=int, default=1, help="Total workers")
    parser.add_argument("--use-ray", action="store_true", help="Use Ray for parallel")
    
    args = parser.parse_args()
    
    # Get coordinator URL from environment or args
    coordinator_url = args.coordinator or os.environ.get(
        "COORDINATOR_URL",
        "http://localhost:5001"
    )
    
    # Create worker config
    worker_id = args.worker_id or f"worker_{os.getpid()}"
    
    config = WorkerConfig(
        worker_id=worker_id,
        coordinator_url=coordinator_url,
        instance_number=args.instance,
        num_workers=args.num_workers,
        use_ray=args.use_ray
    )
    
    # Create and run worker
    worker = create_worker(config)
    worker.run()


if __name__ == "__main__":
    print("=" * 60)
    print("Jupiter Worker - Solana Trading Bot")
    print("=" * 60)
    
    run_worker()
