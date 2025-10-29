#!/usr/bin/env python3
"""
Unit tests for Flash Loan Manager
Tests multi-provider flash loan selection and execution
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
import asyncio

from flash_loan_manager import (
    FlashLoanManager,
    FlashLoanProvider,
    FlashLoanRequest,
    FlashLoanQuote,
    FlashLoanResult,
    AaveV3FlashLoanProvider,
    CurveFlashLoanProvider,
    BalancerV3FlashLoanProvider
)


class TestFlashLoanProviders(unittest.TestCase):
    """Test individual flash loan providers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_web3 = Mock()
        self.mock_web3.eth.gas_price = 50000000000  # 50 gwei
        self.mock_web3.eth.get_transaction_count = Mock(return_value=0)
        
        self.mock_account = Mock()
        self.mock_account.address = "0x1234567890123456789012345678901234567890"
        self.mock_account.key = b"test_key"
    
    def test_aave_v3_fee_percentage(self):
        """Test Aave V3 returns correct fee percentage"""
        provider = AaveV3FlashLoanProvider(
            self.mock_web3,
            "0xAavePool",
            self.mock_account
        )
        
        fee = provider.get_fee_percentage("0xToken", Decimal("1000"))
        self.assertEqual(fee, Decimal('0.0009'))  # 0.09%
    
    def test_curve_fee_percentage(self):
        """Test Curve returns correct fee percentage"""
        provider = CurveFlashLoanProvider(
            self.mock_web3,
            "0xCurvePool",
            self.mock_account
        )
        
        fee = provider.get_fee_percentage("0xToken", Decimal("1000"))
        self.assertEqual(fee, Decimal('0.0004'))  # 0.04%
    
    def test_balancer_v3_fee_percentage(self):
        """Test Balancer V3 returns zero fee"""
        provider = BalancerV3FlashLoanProvider(
            self.mock_web3,
            "0xBalancerVault",
            self.mock_account
        )
        
        fee = provider.get_fee_percentage("0xToken", Decimal("1000"))
        self.assertEqual(fee, Decimal('0'))  # 0%
    
    def test_provider_type_identification(self):
        """Test each provider returns correct type"""
        aave = AaveV3FlashLoanProvider(self.mock_web3, "0xAave", self.mock_account)
        curve = CurveFlashLoanProvider(self.mock_web3, "0xCurve", self.mock_account)
        balancer = BalancerV3FlashLoanProvider(self.mock_web3, "0xBalancer", self.mock_account)
        
        self.assertEqual(aave._get_provider_type(), FlashLoanProvider.AAVE_V3)
        self.assertEqual(curve._get_provider_type(), FlashLoanProvider.CURVE)
        self.assertEqual(balancer._get_provider_type(), FlashLoanProvider.BALANCER_V3)


class TestFlashLoanQuote(unittest.TestCase):
    """Test flash loan quote calculations"""
    
    def test_quote_total_cost(self):
        """Test total cost calculation"""
        quote = FlashLoanQuote(
            provider=FlashLoanProvider.AAVE_V3,
            token="0xToken",
            amount=Decimal("1000"),
            fee=Decimal("9"),  # 0.9
            fee_percentage=Decimal("0.009"),
            available_liquidity=Decimal("10000"),
            gas_estimate=500000
        )
        
        self.assertEqual(quote.total_cost, Decimal("1009"))
    
    def test_quote_effective_cost(self):
        """Test effective cost calculation"""
        quote = FlashLoanQuote(
            provider=FlashLoanProvider.CURVE,
            token="0xToken",
            amount=Decimal("1000"),
            fee=Decimal("4"),
            fee_percentage=Decimal("0.004"),
            available_liquidity=Decimal("10000"),
            gas_estimate=600000
        )
        
        self.assertEqual(quote.effective_cost, Decimal("4"))


class TestFlashLoanManager(unittest.IsolatedAsyncioTestCase):
    """Test Flash Loan Manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_web3 = Mock()
        self.mock_web3.eth.gas_price = 50000000000
        self.mock_web3.eth.get_transaction_count = Mock(return_value=0)
        
        self.mock_account = Mock()
        self.mock_account.address = "0x1234567890123456789012345678901234567890"
        
        self.config = {
            'aave_v3_pool': '0xAavePool',
            'curve_pool': '0xCurvePool',
            'balancer_v3_vault': '0xBalancerVault'
        }
    
    def test_manager_initialization(self):
        """Test manager initializes all configured providers"""
        manager = FlashLoanManager(self.mock_web3, self.mock_account, self.config)
        
        self.assertEqual(len(manager.providers), 3)
        self.assertIn(FlashLoanProvider.AAVE_V3, manager.providers)
        self.assertIn(FlashLoanProvider.CURVE, manager.providers)
        self.assertIn(FlashLoanProvider.BALANCER_V3, manager.providers)
    
    def test_manager_partial_initialization(self):
        """Test manager works with only some providers configured"""
        partial_config = {
            'aave_v3_pool': '0xAavePool',
            'balancer_v3_vault': '0xBalancerVault'
        }
        
        manager = FlashLoanManager(self.mock_web3, self.mock_account, partial_config)
        
        self.assertEqual(len(manager.providers), 2)
        self.assertIn(FlashLoanProvider.AAVE_V3, manager.providers)
        self.assertNotIn(FlashLoanProvider.CURVE, manager.providers)
        self.assertIn(FlashLoanProvider.BALANCER_V3, manager.providers)
    
    async def test_get_best_provider_by_fee(self):
        """Test manager selects provider with lowest fee"""
        manager = FlashLoanManager(self.mock_web3, self.mock_account, self.config)
        
        # Mock get_quote for all providers
        mock_quotes = {
            FlashLoanProvider.AAVE_V3: FlashLoanQuote(
                provider=FlashLoanProvider.AAVE_V3,
                token="0xToken",
                amount=Decimal("1000"),
                fee=Decimal("9"),
                fee_percentage=Decimal("0.009"),
                available_liquidity=Decimal("10000"),
                gas_estimate=800000
            ),
            FlashLoanProvider.CURVE: FlashLoanQuote(
                provider=FlashLoanProvider.CURVE,
                token="0xToken",
                amount=Decimal("1000"),
                fee=Decimal("4"),
                fee_percentage=Decimal("0.004"),
                available_liquidity=Decimal("10000"),
                gas_estimate=600000
            ),
            FlashLoanProvider.BALANCER_V3: FlashLoanQuote(
                provider=FlashLoanProvider.BALANCER_V3,
                token="0xToken",
                amount=Decimal("1000"),
                fee=Decimal("0"),
                fee_percentage=Decimal("0"),
                available_liquidity=Decimal("10000"),
                gas_estimate=700000
            )
        }
        
        # Mock each provider's get_quote method
        for provider_type, provider in manager.providers.items():
            provider.get_quote = AsyncMock(return_value=mock_quotes[provider_type])
        
        # Get best provider
        result = await manager.get_best_provider("0xToken", Decimal("1000"))
        
        self.assertIsNotNone(result)
        best_provider, best_quote = result
        
        # Balancer should be selected (0% fee)
        self.assertEqual(best_provider, FlashLoanProvider.BALANCER_V3)
        self.assertEqual(best_quote.fee, Decimal("0"))
    
    async def test_exclude_active_providers(self):
        """Test manager excludes providers with active loans"""
        manager = FlashLoanManager(self.mock_web3, self.mock_account, self.config)
        
        # Mark Balancer as active
        manager.active_loans[FlashLoanProvider.BALANCER_V3] = True
        
        # Mock get_quote
        mock_quotes = {
            FlashLoanProvider.AAVE_V3: FlashLoanQuote(
                provider=FlashLoanProvider.AAVE_V3,
                token="0xToken",
                amount=Decimal("1000"),
                fee=Decimal("9"),
                fee_percentage=Decimal("0.009"),
                available_liquidity=Decimal("10000"),
                gas_estimate=800000
            ),
            FlashLoanProvider.CURVE: FlashLoanQuote(
                provider=FlashLoanProvider.CURVE,
                token="0xToken",
                amount=Decimal("1000"),
                fee=Decimal("4"),
                fee_percentage=Decimal("0.004"),
                available_liquidity=Decimal("10000"),
                gas_estimate=600000
            )
        }
        
        for provider_type, provider in manager.providers.items():
            if provider_type in mock_quotes:
                provider.get_quote = AsyncMock(return_value=mock_quotes[provider_type])
        
        # Get best provider (should exclude Balancer)
        result = await manager.get_best_provider("0xToken", Decimal("1000"), exclude_active=True)
        
        self.assertIsNotNone(result)
        best_provider, _ = result
        
        # Should be Curve (lowest fee among available)
        self.assertEqual(best_provider, FlashLoanProvider.CURVE)
    
    async def test_execute_flash_loan_success(self):
        """Test successful flash loan execution"""
        manager = FlashLoanManager(self.mock_web3, self.mock_account, self.config)
        
        # Mock successful execution
        mock_result = FlashLoanResult(
            success=True,
            provider=FlashLoanProvider.BALANCER_V3,
            tx_hash="0xabcd1234",
            fee_paid=Decimal("0"),
            gas_used=700000
        )
        
        # Mock provider execution
        manager.providers[FlashLoanProvider.BALANCER_V3].execute_flash_loan = AsyncMock(
            return_value=mock_result
        )
        manager.providers[FlashLoanProvider.BALANCER_V3].get_quote = AsyncMock(
            return_value=FlashLoanQuote(
                provider=FlashLoanProvider.BALANCER_V3,
                token="0xToken",
                amount=Decimal("1000"),
                fee=Decimal("0"),
                fee_percentage=Decimal("0"),
                available_liquidity=Decimal("10000"),
                gas_estimate=700000
            )
        )
        
        # Mock other providers
        for provider_type, provider in manager.providers.items():
            if provider_type != FlashLoanProvider.BALANCER_V3:
                provider.get_quote = AsyncMock(return_value=None)
        
        request = FlashLoanRequest(
            token="0xToken",
            amount=Decimal("1000"),
            callback_data=b"test_data",
            max_fee_percentage=Decimal("0.01")
        )
        
        result = await manager.execute_flash_loan(request)
        
        self.assertTrue(result.success)
        self.assertEqual(result.provider, FlashLoanProvider.BALANCER_V3)
        self.assertEqual(result.tx_hash, "0xabcd1234")
    
    async def test_execute_multiple_flash_loans(self):
        """Test executing multiple flash loans simultaneously"""
        manager = FlashLoanManager(self.mock_web3, self.mock_account, self.config)
        
        # Create mock results for each provider
        mock_results = [
            FlashLoanResult(
                success=True,
                provider=FlashLoanProvider.AAVE_V3,
                tx_hash="0xaave",
                fee_paid=Decimal("9"),
                gas_used=800000
            ),
            FlashLoanResult(
                success=True,
                provider=FlashLoanProvider.CURVE,
                tx_hash="0xcurve",
                fee_paid=Decimal("4"),
                gas_used=600000
            ),
            FlashLoanResult(
                success=True,
                provider=FlashLoanProvider.BALANCER_V3,
                tx_hash="0xbalancer",
                fee_paid=Decimal("0"),
                gas_used=700000
            )
        ]
        
        # Mock execute_flash_loan to return specific results
        call_count = 0
        async def mock_execute(request):
            nonlocal call_count
            result = mock_results[call_count]
            call_count += 1
            return result
        
        manager.execute_flash_loan = mock_execute
        
        # Create three requests
        requests = [
            FlashLoanRequest(
                token="0xToken1",
                amount=Decimal("1000"),
                callback_data=b"data1"
            ),
            FlashLoanRequest(
                token="0xToken2",
                amount=Decimal("2000"),
                callback_data=b"data2"
            ),
            FlashLoanRequest(
                token="0xToken3",
                amount=Decimal("3000"),
                callback_data=b"data3"
            )
        ]
        
        results = await manager.execute_multiple_flash_loans(requests)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(results[0].provider, FlashLoanProvider.AAVE_V3)
        self.assertEqual(results[1].provider, FlashLoanProvider.CURVE)
        self.assertEqual(results[2].provider, FlashLoanProvider.BALANCER_V3)
    
    def test_get_provider_status(self):
        """Test getting provider status"""
        manager = FlashLoanManager(self.mock_web3, self.mock_account, self.config)
        
        # Mark one provider as active
        manager.active_loans[FlashLoanProvider.AAVE_V3] = True
        
        status = manager.get_provider_status()
        
        self.assertEqual(len(status), 3)
        self.assertFalse(status['aave_v3']['available'])
        self.assertTrue(status['aave_v3']['active_loan'])
        self.assertTrue(status['curve']['available'])
        self.assertFalse(status['curve']['active_loan'])


class TestFlashLoanRequest(unittest.TestCase):
    """Test flash loan request validation"""
    
    def test_request_creation(self):
        """Test creating a flash loan request"""
        request = FlashLoanRequest(
            token="0xToken",
            amount=Decimal("1000"),
            callback_data=b"test_data",
            max_fee_percentage=Decimal("0.005")
        )
        
        self.assertEqual(request.token, "0xToken")
        self.assertEqual(request.amount, Decimal("1000"))
        self.assertEqual(request.callback_data, b"test_data")
        self.assertEqual(request.max_fee_percentage, Decimal("0.005"))
    
    def test_request_default_max_fee(self):
        """Test default max fee percentage"""
        request = FlashLoanRequest(
            token="0xToken",
            amount=Decimal("1000"),
            callback_data=b"test_data"
        )
        
        self.assertEqual(request.max_fee_percentage, Decimal("0.01"))  # 1% default


class TestFlashLoanResult(unittest.TestCase):
    """Test flash loan result handling"""
    
    def test_successful_result(self):
        """Test successful flash loan result"""
        result = FlashLoanResult(
            success=True,
            provider=FlashLoanProvider.CURVE,
            tx_hash="0xabcd1234",
            fee_paid=Decimal("4"),
            gas_used=600000
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.provider, FlashLoanProvider.CURVE)
        self.assertIsNotNone(result.tx_hash)
        self.assertIsNone(result.error_message)
    
    def test_failed_result(self):
        """Test failed flash loan result"""
        result = FlashLoanResult(
            success=False,
            provider=FlashLoanProvider.AAVE_V3,
            tx_hash=None,
            fee_paid=Decimal("0"),
            gas_used=0,
            error_message="Insufficient liquidity"
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.tx_hash)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.error_message, "Insufficient liquidity")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFlashLoanProviders))
    suite.addTests(loader.loadTestsFromTestCase(TestFlashLoanQuote))
    suite.addTests(loader.loadTestsFromTestCase(TestFlashLoanManager))
    suite.addTests(loader.loadTestsFromTestCase(TestFlashLoanRequest))
    suite.addTests(loader.loadTestsFromTestCase(TestFlashLoanResult))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
