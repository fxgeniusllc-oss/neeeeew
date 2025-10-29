#!/usr/bin/env python3
"""
Flash Loan Manager - Multi-Provider Flash Loan System
Supports Curve Pool, Aave V3 Pool, and Balancer Vault v3
Dynamic provider selection based on fees and liquidity availability
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from web3 import Web3
from eth_account import Account


logger = logging.getLogger(__name__)


class FlashLoanProvider(Enum):
    """Supported flash loan providers"""
    AAVE_V3 = "aave_v3"
    CURVE = "curve"
    BALANCER_V3 = "balancer_v3"


@dataclass
class FlashLoanQuote:
    """Quote for a flash loan from a provider"""
    provider: FlashLoanProvider
    token: str
    amount: Decimal
    fee: Decimal
    fee_percentage: Decimal
    available_liquidity: Decimal
    gas_estimate: int
    
    @property
    def total_cost(self) -> Decimal:
        """Total cost including fee"""
        return self.amount + self.fee
    
    @property
    def effective_cost(self) -> Decimal:
        """Effective cost including gas (in token terms)"""
        # Simplified: actual implementation would convert gas to token value
        return self.fee


@dataclass
class FlashLoanRequest:
    """Flash loan request parameters"""
    token: str
    amount: Decimal
    callback_data: bytes
    max_fee_percentage: Decimal = Decimal('0.01')  # 1% default max fee


@dataclass
class FlashLoanResult:
    """Result of a flash loan execution"""
    success: bool
    provider: FlashLoanProvider
    tx_hash: Optional[str]
    fee_paid: Decimal
    gas_used: int
    error_message: Optional[str] = None


class BaseFlashLoanProvider(ABC):
    """Abstract base class for flash loan providers"""
    
    def __init__(self, web3: Web3, contract_address: str, account: Account):
        self.web3 = web3
        self.contract_address = contract_address
        self.account = account
        self.contract = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def get_fee_percentage(self, token: str, amount: Decimal) -> Decimal:
        """Get fee percentage for a flash loan"""
        pass
    
    @abstractmethod
    def get_available_liquidity(self, token: str) -> Decimal:
        """Get available liquidity for a token"""
        pass
    
    @abstractmethod
    async def execute_flash_loan(self, request: FlashLoanRequest) -> FlashLoanResult:
        """Execute a flash loan"""
        pass
    
    async def get_quote(self, token: str, amount: Decimal) -> Optional[FlashLoanQuote]:
        """Get a quote for a flash loan"""
        try:
            fee_pct = self.get_fee_percentage(token, amount)
            fee = amount * fee_pct
            liquidity = self.get_available_liquidity(token)
            
            if liquidity < amount:
                self.logger.warning(f"Insufficient liquidity: {liquidity} < {amount}")
                return None
            
            # Estimate gas (these are approximate values)
            gas_estimate = self._estimate_gas()
            
            return FlashLoanQuote(
                provider=self._get_provider_type(),
                token=token,
                amount=amount,
                fee=fee,
                fee_percentage=fee_pct,
                available_liquidity=liquidity,
                gas_estimate=gas_estimate
            )
        except Exception as e:
            self.logger.error(f"Error getting quote: {e}")
            return None
    
    @abstractmethod
    def _get_provider_type(self) -> FlashLoanProvider:
        """Get the provider type"""
        pass
    
    def _estimate_gas(self) -> int:
        """Estimate gas for flash loan transaction"""
        # Override in subclasses for provider-specific estimates
        return 500000


class AaveV3FlashLoanProvider(BaseFlashLoanProvider):
    """Aave V3 Pool flash loan provider"""
    
    def __init__(self, web3: Web3, pool_address: str, account: Account):
        super().__init__(web3, pool_address, account)
        self.contract = self._initialize_contract()
    
    def _initialize_contract(self):
        """Initialize Aave V3 Pool contract"""
        abi = [
            {
                "inputs": [
                    {"name": "receiverAddress", "type": "address"},
                    {"name": "assets", "type": "address[]"},
                    {"name": "amounts", "type": "uint256[]"},
                    {"name": "interestRateModes", "type": "uint256[]"},
                    {"name": "onBehalfOf", "type": "address"},
                    {"name": "params", "type": "bytes"},
                    {"name": "referralCode", "type": "uint16"}
                ],
                "name": "flashLoan",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "asset", "type": "address"}],
                "name": "getReserveData",
                "outputs": [
                    {
                        "components": [
                            {"name": "availableLiquidity", "type": "uint256"},
                            {"name": "totalStableDebt", "type": "uint256"},
                            {"name": "totalVariableDebt", "type": "uint256"}
                        ],
                        "type": "tuple"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        return self.web3.eth.contract(address=self.contract_address, abi=abi)
    
    def get_fee_percentage(self, token: str, amount: Decimal) -> Decimal:
        """Aave V3 charges 0.09% fee"""
        return Decimal('0.0009')
    
    def get_available_liquidity(self, token: str) -> Decimal:
        """Get available liquidity from Aave V3 pool"""
        try:
            reserve_data = self.contract.functions.getReserveData(token).call()
            liquidity = Decimal(reserve_data[0]) / Decimal(1e18)
            return liquidity
        except Exception as e:
            self.logger.error(f"Error getting Aave liquidity: {e}")
            return Decimal('0')
    
    async def execute_flash_loan(self, request: FlashLoanRequest) -> FlashLoanResult:
        """Execute Aave V3 flash loan"""
        try:
            self.logger.info(f"Executing Aave V3 flash loan: {request.amount} {request.token}")
            
            # Build transaction
            tx = self.contract.functions.flashLoan(
                self.account.address,  # receiverAddress
                [request.token],  # assets
                [int(request.amount * Decimal(1e18))],  # amounts
                [0],  # interestRateModes (0 = no debt)
                self.account.address,  # onBehalfOf
                request.callback_data,  # params
                0  # referralCode
            ).build_transaction({
                'from': self.account.address,
                'gas': 800000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
            
            # Wait for receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            fee_paid = request.amount * self.get_fee_percentage(request.token, request.amount)
            
            return FlashLoanResult(
                success=receipt['status'] == 1,
                provider=FlashLoanProvider.AAVE_V3,
                tx_hash=tx_hash.hex(),
                fee_paid=fee_paid,
                gas_used=receipt['gasUsed']
            )
            
        except Exception as e:
            self.logger.error(f"Aave V3 flash loan error: {e}")
            return FlashLoanResult(
                success=False,
                provider=FlashLoanProvider.AAVE_V3,
                tx_hash=None,
                fee_paid=Decimal('0'),
                gas_used=0,
                error_message=str(e)
            )
    
    def _get_provider_type(self) -> FlashLoanProvider:
        return FlashLoanProvider.AAVE_V3
    
    def _estimate_gas(self) -> int:
        return 800000


class CurveFlashLoanProvider(BaseFlashLoanProvider):
    """Curve Pool flash loan provider"""
    
    def __init__(self, web3: Web3, pool_address: str, account: Account):
        super().__init__(web3, pool_address, account)
        self.contract = self._initialize_contract()
    
    def _initialize_contract(self):
        """Initialize Curve Pool contract"""
        abi = [
            {
                "inputs": [
                    {"name": "receiver", "type": "address"},
                    {"name": "token", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "data", "type": "bytes"}
                ],
                "name": "flashLoan",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "i", "type": "uint256"}],
                "name": "balances",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "i", "type": "uint256"}],
                "name": "coins",
                "outputs": [{"name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        return self.web3.eth.contract(address=self.contract_address, abi=abi)
    
    def get_fee_percentage(self, token: str, amount: Decimal) -> Decimal:
        """Curve typically charges 0.04% fee (varies by pool)"""
        return Decimal('0.0004')
    
    def get_available_liquidity(self, token: str) -> Decimal:
        """Get available liquidity from Curve pool"""
        try:
            # Try to find token in pool coins
            for i in range(8):  # Curve pools typically have up to 8 coins
                try:
                    coin = self.contract.functions.coins(i).call()
                    if coin.lower() == token.lower():
                        balance = self.contract.functions.balances(i).call()
                        return Decimal(balance) / Decimal(1e18)
                except:
                    break
            return Decimal('0')
        except Exception as e:
            self.logger.error(f"Error getting Curve liquidity: {e}")
            return Decimal('0')
    
    async def execute_flash_loan(self, request: FlashLoanRequest) -> FlashLoanResult:
        """Execute Curve flash loan"""
        try:
            self.logger.info(f"Executing Curve flash loan: {request.amount} {request.token}")
            
            # Build transaction
            tx = self.contract.functions.flashLoan(
                self.account.address,  # receiver
                request.token,  # token
                int(request.amount * Decimal(1e18)),  # amount
                request.callback_data  # data
            ).build_transaction({
                'from': self.account.address,
                'gas': 600000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
            
            # Wait for receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            fee_paid = request.amount * self.get_fee_percentage(request.token, request.amount)
            
            return FlashLoanResult(
                success=receipt['status'] == 1,
                provider=FlashLoanProvider.CURVE,
                tx_hash=tx_hash.hex(),
                fee_paid=fee_paid,
                gas_used=receipt['gasUsed']
            )
            
        except Exception as e:
            self.logger.error(f"Curve flash loan error: {e}")
            return FlashLoanResult(
                success=False,
                provider=FlashLoanProvider.CURVE,
                tx_hash=None,
                fee_paid=Decimal('0'),
                gas_used=0,
                error_message=str(e)
            )
    
    def _get_provider_type(self) -> FlashLoanProvider:
        return FlashLoanProvider.CURVE
    
    def _estimate_gas(self) -> int:
        return 600000


class BalancerV3FlashLoanProvider(BaseFlashLoanProvider):
    """Balancer Vault v3 flash loan provider"""
    
    def __init__(self, web3: Web3, vault_address: str, account: Account):
        super().__init__(web3, vault_address, account)
        self.contract = self._initialize_contract()
    
    def _initialize_contract(self):
        """Initialize Balancer Vault v3 contract"""
        abi = [
            {
                "inputs": [
                    {"name": "recipient", "type": "address"},
                    {"name": "tokens", "type": "address[]"},
                    {"name": "amounts", "type": "uint256[]"},
                    {"name": "userData", "type": "bytes"}
                ],
                "name": "flashLoan",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "token", "type": "address"}],
                "name": "getPoolTokenInfo",
                "outputs": [
                    {"name": "cash", "type": "uint256"},
                    {"name": "managed", "type": "uint256"},
                    {"name": "lastChangeBlock", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        return self.web3.eth.contract(address=self.contract_address, abi=abi)
    
    def get_fee_percentage(self, token: str, amount: Decimal) -> Decimal:
        """Balancer V3 charges 0% fee (only gas costs)"""
        return Decimal('0')
    
    def get_available_liquidity(self, token: str) -> Decimal:
        """Get available liquidity from Balancer Vault"""
        try:
            pool_info = self.contract.functions.getPoolTokenInfo(token).call()
            cash = Decimal(pool_info[0]) / Decimal(1e18)
            return cash
        except Exception as e:
            self.logger.error(f"Error getting Balancer liquidity: {e}")
            return Decimal('0')
    
    async def execute_flash_loan(self, request: FlashLoanRequest) -> FlashLoanResult:
        """Execute Balancer V3 flash loan"""
        try:
            self.logger.info(f"Executing Balancer V3 flash loan: {request.amount} {request.token}")
            
            # Build transaction
            tx = self.contract.functions.flashLoan(
                self.account.address,  # recipient
                [request.token],  # tokens
                [int(request.amount * Decimal(1e18))],  # amounts
                request.callback_data  # userData
            ).build_transaction({
                'from': self.account.address,
                'gas': 700000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
            
            # Wait for receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            fee_paid = request.amount * self.get_fee_percentage(request.token, request.amount)
            
            return FlashLoanResult(
                success=receipt['status'] == 1,
                provider=FlashLoanProvider.BALANCER_V3,
                tx_hash=tx_hash.hex(),
                fee_paid=fee_paid,
                gas_used=receipt['gasUsed']
            )
            
        except Exception as e:
            self.logger.error(f"Balancer V3 flash loan error: {e}")
            return FlashLoanResult(
                success=False,
                provider=FlashLoanProvider.BALANCER_V3,
                tx_hash=None,
                fee_paid=Decimal('0'),
                gas_used=0,
                error_message=str(e)
            )
    
    def _get_provider_type(self) -> FlashLoanProvider:
        return FlashLoanProvider.BALANCER_V3
    
    def _estimate_gas(self) -> int:
        return 700000


class FlashLoanManager:
    """
    Manages multiple flash loan providers and selects the best option
    Supports simultaneous flash loans from different providers (max 1 per provider)
    """
    
    def __init__(self, web3: Web3, account: Account, config: Dict):
        self.web3 = web3
        self.account = account
        self.config = config
        self.providers: Dict[FlashLoanProvider, BaseFlashLoanProvider] = {}
        self.active_loans: Dict[FlashLoanProvider, bool] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize providers from config
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured flash loan providers"""
        # Aave V3
        if 'aave_v3_pool' in self.config:
            self.providers[FlashLoanProvider.AAVE_V3] = AaveV3FlashLoanProvider(
                self.web3,
                self.config['aave_v3_pool'],
                self.account
            )
            self.active_loans[FlashLoanProvider.AAVE_V3] = False
            self.logger.info("Initialized Aave V3 flash loan provider")
        
        # Curve
        if 'curve_pool' in self.config:
            self.providers[FlashLoanProvider.CURVE] = CurveFlashLoanProvider(
                self.web3,
                self.config['curve_pool'],
                self.account
            )
            self.active_loans[FlashLoanProvider.CURVE] = False
            self.logger.info("Initialized Curve flash loan provider")
        
        # Balancer V3
        if 'balancer_v3_vault' in self.config:
            self.providers[FlashLoanProvider.BALANCER_V3] = BalancerV3FlashLoanProvider(
                self.web3,
                self.config['balancer_v3_vault'],
                self.account
            )
            self.active_loans[FlashLoanProvider.BALANCER_V3] = False
            self.logger.info("Initialized Balancer V3 flash loan provider")
        
        self.logger.info(f"Flash Loan Manager initialized with {len(self.providers)} providers")
    
    async def get_best_provider(
        self, 
        token: str, 
        amount: Decimal,
        exclude_active: bool = True
    ) -> Optional[Tuple[FlashLoanProvider, FlashLoanQuote]]:
        """
        Find the best provider for a flash loan based on fees and liquidity
        
        Args:
            token: Token address
            amount: Amount to borrow
            exclude_active: Whether to exclude providers with active loans
        
        Returns:
            Tuple of (provider, quote) or None if no provider available
        """
        quotes: List[Tuple[FlashLoanProvider, FlashLoanQuote]] = []
        
        for provider_type, provider in self.providers.items():
            # Skip if provider has active loan and we want to exclude active
            if exclude_active and self.active_loans.get(provider_type, False):
                self.logger.debug(f"Skipping {provider_type.value} - active loan")
                continue
            
            quote = await provider.get_quote(token, amount)
            if quote:
                quotes.append((provider_type, quote))
        
        if not quotes:
            self.logger.warning(f"No available providers for {amount} {token}")
            return None
        
        # Sort by effective cost (fee + gas consideration)
        quotes.sort(key=lambda x: x[1].effective_cost)
        
        best_provider, best_quote = quotes[0]
        self.logger.info(
            f"Best provider: {best_provider.value} with fee {best_quote.fee_percentage:.4%}"
        )
        
        return best_provider, best_quote
    
    async def execute_flash_loan(
        self, 
        request: FlashLoanRequest,
        preferred_provider: Optional[FlashLoanProvider] = None
    ) -> FlashLoanResult:
        """
        Execute a flash loan with automatic provider selection
        
        Args:
            request: Flash loan request parameters
            preferred_provider: Optional preferred provider, will fallback if unavailable
        
        Returns:
            FlashLoanResult with execution details
        """
        provider_type = preferred_provider
        
        # If no preferred provider or preferred is unavailable, find best
        if not provider_type or self.active_loans.get(provider_type, False):
            best = await self.get_best_provider(
                request.token, 
                request.amount,
                exclude_active=True
            )
            if not best:
                return FlashLoanResult(
                    success=False,
                    provider=preferred_provider or FlashLoanProvider.AAVE_V3,
                    tx_hash=None,
                    fee_paid=Decimal('0'),
                    gas_used=0,
                    error_message="No available providers"
                )
            provider_type, quote = best
            
            # Check fee against max
            if quote.fee_percentage > request.max_fee_percentage:
                return FlashLoanResult(
                    success=False,
                    provider=provider_type,
                    tx_hash=None,
                    fee_paid=Decimal('0'),
                    gas_used=0,
                    error_message=f"Fee too high: {quote.fee_percentage:.4%} > {request.max_fee_percentage:.4%}"
                )
        
        provider = self.providers[provider_type]
        
        # Mark provider as active
        self.active_loans[provider_type] = True
        
        try:
            # Execute flash loan
            result = await provider.execute_flash_loan(request)
            return result
        finally:
            # Mark provider as inactive
            self.active_loans[provider_type] = False
    
    async def execute_multiple_flash_loans(
        self, 
        requests: List[FlashLoanRequest]
    ) -> List[FlashLoanResult]:
        """
        Execute multiple flash loans simultaneously (max 1 per provider)
        
        Args:
            requests: List of flash loan requests
        
        Returns:
            List of FlashLoanResults
        """
        import asyncio
        
        if len(requests) > len(self.providers):
            self.logger.warning(
                f"Cannot execute {len(requests)} simultaneous loans with only {len(self.providers)} providers"
            )
        
        # Execute loans concurrently
        tasks = [self.execute_flash_loan(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(FlashLoanResult(
                    success=False,
                    provider=FlashLoanProvider.AAVE_V3,  # Default
                    tx_hash=None,
                    fee_paid=Decimal('0'),
                    gas_used=0,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_provider_status(self) -> Dict:
        """Get status of all providers"""
        status = {}
        for provider_type, provider in self.providers.items():
            status[provider_type.value] = {
                'available': not self.active_loans.get(provider_type, False),
                'active_loan': self.active_loans.get(provider_type, False),
                'contract_address': provider.contract_address
            }
        return status
