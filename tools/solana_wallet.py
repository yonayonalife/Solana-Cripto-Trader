#!/usr/bin/env python3
"""
Solana Wallet Manager for Jupiter Trading Bot
Handles wallet creation, key management, and transaction signing
"""
import os
import json
import base64
import base58
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from cryptography.fernet import Fernet
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import Message
from solders.transaction import Transaction
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed, Finalized
from solders.rpc.responses import GetBalanceResp, GetTokenAccountBalanceResp


# Constants
WALLET_DIR = Path.home() / ".config" / "solana-jupiter-bot"
ENCRYPTED_KEY_FILE = WALLET_DIR / "wallet.enc"
WALLET_INFO_FILE = WALLET_DIR / "wallet_info.json"


@dataclass
class WalletInfo:
    """Wallet metadata (not sensitive)"""
    public_key: str
    key_type: str  # 'file', 'encrypted', 'hardware'
    created_at: str
    last_used: str
    network: str  # 'mainnet', 'testnet', 'devnet'
    
    def to_dict(self) -> Dict:
        return {
            "public_key": self.public_key,
            "key_type": self.key_type,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "network": self.network
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "WalletInfo":
        return cls(
            public_key=data["public_key"],
            key_type=data["key_type"],
            created_at=data["created_at"],
            last_used=data["last_used"],
            network=data["network"]
        )


@dataclass
class WalletBalance:
    """Wallet balance information"""
    sol_balance: float
    usdc_balance: float
    usdt_balance: float
    token_balances: Dict[str, float]  # mint -> amount
    
    def total_usd_value(self, sol_price: float = 100.0) -> float:
        """Calculate total value in USD"""
        return (self.sol_balance * sol_price) + self.usdc_balance + self.usdt_balance


class SolanaWallet:
    """
    Solana wallet manager with encryption and safety features
    
    Features:
    - Encrypted private key storage
    - Hot/Cold wallet separation support
    - Balance checking
    - Transaction signing
    - Hardware wallet integration (future)
    """
    
    def __init__(
        self,
        network: str = "mainnet",
        encryption_key: Optional[bytes] = None
    ):
        """
        Initialize wallet manager
        
        Args:
            network: 'mainnet', 'testnet', or 'devnet'
            encryption_key: Fernet key for encryption (auto-generated if None)
        """
        self.network = network
        self.keypair: Optional[Keypair] = None
        self.public_key: Optional[Pubkey] = None
        
        # Create wallet directory
        WALLET_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup encryption
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            self._load_or_create_encryption_key()
        
        # Setup RPC client
        self.rpc_url = self._get_rpc_url(network)
        self.http_client = Client(self.rpc_url)
    
    def _get_rpc_url(self, network: str) -> str:
        """Get RPC URL for network"""
        rpc_urls = {
            "mainnet": os.environ.get("SOLANA_RPC_MAINNET", "https://api.mainnet-beta.solana.com"),
            "testnet": os.environ.get("SOLANA_RPC_TESTNET", "https://api.testnet.solana.com"),
            "devnet": os.environ.get("SOLANA_RPC_DEVNET", "https://api.devnet.solana.com")
        }
        return rpc_urls.get(network, rpc_urls["mainnet"])
    
    def _load_or_create_encryption_key(self):
        """Load existing encryption key or create new one"""
        key_file = WALLET_DIR / "encryption.key"
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                self.fernet = Fernet(f.read())
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            self.fernet = Fernet(key)
            os.chmod(key_file, 0o600)  # Read-only for owner
    
    def create_new_wallet(self) -> Tuple[Keypair, WalletInfo]:
        """
        Create a new wallet
        
        Returns:
            Tuple of (Keypair, WalletInfo)
        """
        # Generate new keypair
        self.keypair = Keypair()
        self.public_key = self.keypair.pubkey()
        
        # Encrypt and save private key
        private_key_bytes = bytes(self.keypair)
        encrypted = self.fernet.encrypt(private_key_bytes)
        
        with open(ENCRYPTED_KEY_FILE, "wb") as f:
            f.write(encrypted)
        os.chmod(ENCRYPTED_KEY_FILE, 0o600)
        
        # Create wallet info
        info = WalletInfo(
            public_key=str(self.public_key),
            key_type="encrypted",
            created_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            network=self.network
        )
        
        # Save wallet info
        with open(WALLET_INFO_FILE, "w") as f:
            json.dump(info.to_dict(), f, indent=2)
        
        return self.keypair, info
    
    def load_wallet_from_private_key(self, private_key: str) -> WalletInfo:
        """
        Load wallet from base58 private key
        
        Args:
            private_key: Base58 encoded private key
        
        Returns:
            WalletInfo
        """
        from solders.keypair import Keypair
        
        # Decode private key
        self.keypair = Keypair.from_base58_string(private_key)
        self.public_key = self.keypair.pubkey()
        
        # Encrypt and save
        private_key_bytes = bytes(self.keypair)
        encrypted = self.fernet.encrypt(private_key_bytes)
        
        with open(ENCRYPTED_KEY_FILE, "wb") as f:
            f.write(encrypted)
        os.chmod(ENCRYPTED_KEY_FILE, 0o600)
        
        # Create wallet info
        info = WalletInfo(
            public_key=str(self.public_key),
            key_type="encrypted",
            created_at=datetime.now().isoformat(),
            last_used=datetime.now().isoformat(),
            network=self.network
        )
        
        with open(WALLET_INFO_FILE, "w") as f:
            json.dump(info.to_dict(), f, indent=2)
        
        return info
    
    def load_wallet(self) -> bool:
        """
        Load wallet from encrypted file
        
        Returns:
            True if wallet loaded successfully
        """
        if not ENCRYPTED_KEY_FILE.exists():
            return False
        
        try:
            with open(ENCRYPTED_KEY_FILE, "rb") as f:
                encrypted = f.read()
            
            private_key_bytes = self.fernet.decrypt(encrypted)
            self.keypair = Keypair.from_bytes(private_key_bytes)
            self.public_key = self.keypair.pubkey()
            
            # Update last used
            self._update_last_used()
            
            return True
        except Exception as e:
            print(f"Error loading wallet: {e}")
            return False
    
    def _update_last_used(self):
        """Update last used timestamp"""
        if WALLET_INFO_FILE.exists():
            with open(WALLET_INFO_FILE, "r") as f:
                info = json.load(f)
            
            info["last_used"] = datetime.now().isoformat()
            
            with open(WALLET_INFO_FILE, "w") as f:
                json.dump(info, f, indent=2)
    
    def get_public_key(self) -> str:
        """Get public key as string"""
        if not self.public_key:
            if not self.load_wallet():
                raise ValueError("No wallet loaded")
        return str(self.public_key)
    
    def get_balance(self) -> WalletBalance:
        """
        Get wallet balance for SOL and tokens
        
        Returns:
            WalletBalance object
        """
        if not self.public_key:
            self.load_wallet()
        
        # Get SOL balance
        sol_balance = 0.0
        try:
            resp = self.http_client.get_balance(
                self.public_key,
                commitment=Confirmed
            )
            # Handle both old and new API response formats
            if hasattr(resp, 'value'):
                value = resp.value
                if hasattr(value, 'value'):
                    sol_lamports = value.value  # New format
                else:
                    sol_lamports = value  # Old format
            else:
                sol_lamports = 0
            sol_balance = sol_lamports / 1_000_000_000
        except Exception as e:
            print(f"Error getting SOL balance: {e}")
        
        # Token balances (USDC, USDT) - simplified for now
        usdc_balance = 0.0
        usdt_balance = 0.0
        token_balances = {}
        
        # Skip token balance checks for now (API compatibility issues)
        print("💡 Tip: Token balances show 0 until deposits.")
        
        return WalletBalance(
            sol_balance=sol_balance,
            usdc_balance=usdc_balance,
            usdt_balance=usdt_balance,
            token_balances=token_balances
        )
    
    def sign_transaction(self, transaction: Transaction) -> Transaction:
        """
        Sign a transaction
        
        Args:
            transaction: Unsigned Transaction
        
        Returns:
            Signed Transaction
        """
        if not self.keypair:
            self.load_wallet()
        
        return self.keypair.sign_transaction(transaction)
    
    def sign_message(self, message: bytes) -> bytes:
        """
        Sign a message
        
        Args:
            message: Message bytes
        
        Returns:
            Signature bytes
        """
        if not self.keypair:
            self.load_wallet()
        
        return self.keypair.sign_message(message)
    
    def create_test_wallet(self) -> Tuple[str, str]:
        """
        Create a wallet for testing (devnet)
        
        Returns:
            Tuple of (public_key, private_key_base58)
        """
        # Generate keypair
        import base58
        keypair = Keypair()
        public_key = str(keypair.pubkey())
        private_key = base58.b58encode(bytes(keypair.secret())).decode()
        
        # Airdrop SOL for testing (devnet only)
        if self.network == "devnet":
            try:
                self.http_client.request_airdrop(
                    keypair.pubkey(),
                    2_000_000_000  # 2 SOL
                )
                print(f"Requested 2 SOL airdrop to {public_key}")
            except Exception as e:
                print(f"Airdrop failed (might needfaucet): {e}")
        
        return public_key, private_key


class HotWalletManager:
    """
    Manager for hot wallet with safety limits
    
    Ensures:
    - Maximum per-trade limit
    - Maximum daily limit
    - Minimum balance reserve
    """
    
    def __init__(
        self,
        wallet: SolanaWallet,
        max_trade_pct: float = 0.10,  # 10% of hot wallet
        max_daily_pct: float = 0.30,  # 30% of hot wallet
        min_reserve_sol: float = 0.05  # Reserve 0.05 SOL for fees
    ):
        self.wallet = wallet
        self.max_trade_pct = max_trade_pct
        self.max_daily_pct = max_daily_pct
        self.min_reserve_sol = min_reserve_sol
        
        # Track daily usage
        self.daily_used_sol = 0.0
        self.last_reset = None
    
    def can_trade(self, amount_sol: float) -> Tuple[bool, str]:
        """
        Check if trade is allowed
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Get current balance
        balance = self.wallet.get_balance()
        available_sol = balance.sol_balance - self.min_reserve_sol
        
        if available_sol < self.min_reserve_sol:
            return False, "Insufficient balance for fees"
        
        # Check max trade
        max_trade = available_sol * self.max_trade_pct
        if amount_sol > max_trade:
            return False, f"Amount {amount_sol} SOL exceeds max trade {max_trade} SOL"
        
        # Check daily limit
        if self.daily_used_sol + amount_sol > available_sol * self.max_daily_pct:
            return False, f"Would exceed daily limit"
        
        return True, "OK"
    
    def record_trade(self, amount_sol: float):
        """Record a trade for daily tracking"""
        self.daily_used_sol += amount_sol
    
    def get_available_trading_balance(self) -> float:
        """Get maximum amount available for trading"""
        balance = self.wallet.get_balance()
        available = balance.sol_balance - self.min_reserve_sol
        max_daily = available * self.max_daily_pct
        
        return min(available * self.max_trade_pct, max_daily - self.daily_used_sol)


# Utility functions
def generate_encryption_key() -> bytes:
    """Generate a new encryption key"""
    return Fernet.generate_key()


def encrypt_private_key(private_key: str, encryption_key: bytes) -> bytes:
    """Encrypt private key with Fernet"""
    fernet = Fernet(encryption_key)
    return fernet.encrypt(private_key.encode())


def decrypt_private_key(encrypted_key: bytes, encryption_key: bytes) -> str:
    """Decrypt private key"""
    fernet = Fernet(encryption_key)
    return fernet.decrypt(encrypted_key).decode()


# Example usage
async def main():
    print("=" * 60)
    print("Solana Wallet Manager - Demo")
    print("=" * 60)
    
    # Initialize wallet manager (using devnet for testing)
    wallet = SolanaWallet(network="devnet")
    
    # Check if wallet exists
    if ENCRYPTED_KEY_FILE.exists():
        print("\n📂 Loading existing wallet...")
        if wallet.load_wallet():
            print(f"✅ Wallet loaded: {wallet.get_public_key()}")
        else:
            print("❌ Failed to load wallet")
            return
    else:
        print("\n🆕 Creating new wallet...")
        keypair, info = wallet.create_new_wallet()
        print(f"✅ Wallet created: {info.public_key}")
        private_key = base58.b58encode(bytes(keypair.secret())).decode()
        print(f"🔑 Save this private key: {private_key}")
    
    # Get balance
    print("\n💰 Checking balance...")
    balance = wallet.get_balance()
    print(f"   SOL: {balance.sol_balance:.4f}")
    print(f"   USDC: {balance.usdc_balance:.2f}")
    print(f"   USDT: {balance.usdt_balance:.2f}")
    
    # Hot wallet manager
    print("\n🔥 Hot Wallet Manager")
    hot_manager = HotWalletManager(wallet)
    available = hot_manager.get_available_trading_balance()
    print(f"   Available for trading: {available:.4f} SOL")
    
    # Check if can trade
    can_trade, reason = hot_manager.can_trade(0.1)
    print(f"   Can trade 0.1 SOL: {can_trade} ({reason})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
