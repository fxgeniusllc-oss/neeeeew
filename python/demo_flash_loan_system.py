#!/usr/bin/env python3
"""
Flash Loan Manager Demo
Demonstrates the multi-provider flash loan system
"""

import asyncio
from decimal import Decimal
from web3 import Web3
from eth_account import Account

# Import flash loan components
from flash_loan_manager import (
    FlashLoanManager,
    FlashLoanProvider,
    FlashLoanRequest,
)


async def demo_flash_loan_system():
    """Demonstrate flash loan system capabilities"""
    
    print("🚀 Flash Loan Multi-Provider System Demo")
    print("=" * 60)
    
    # Setup Web3 (using a public RPC for demo)
    w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
    
    # Create a demo account (DO NOT use in production without secure key management)
    private_key = "0x" + "0" * 64  # Demo key - replace with actual key
    account = Account.from_key(private_key)
    
    print(f"\n📍 Connected to Polygon network: {w3.is_connected()}")
    print(f"📍 Account address: {account.address}")
    
    # Configure flash loan providers
    config = {
        'aave_v3_pool': '0x794a61458eD90ABD2294aB7e655BC0fD30C4D0c8',
        'balancer_v3_vault': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
    }
    
    # Initialize Flash Loan Manager
    print("\n🔧 Initializing Flash Loan Manager...")
    manager = FlashLoanManager(w3, account, config)
    
    print(f"✅ Initialized with {len(manager.providers)} providers:")
    for provider_type in manager.providers.keys():
        print(f"   - {provider_type.value}")
    
    # Get provider status
    print("\n📊 Provider Status:")
    status = manager.get_provider_status()
    for provider, info in status.items():
        availability = "✅ Available" if info['available'] else "❌ Busy"
        print(f"   {provider.upper():20s} {availability:15s} Contract: {info['contract_address']}")
    
    # Demo: Find best provider for a flash loan
    print("\n🔍 Finding best provider for 1000 USDC flash loan...")
    
    # USDC token address on Polygon
    usdc_token = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    amount = Decimal("1000")
    
    try:
        best = await manager.get_best_provider(usdc_token, amount)
        
        if best:
            provider, quote = best
            print(f"\n✨ Best Provider: {provider.value}")
            print(f"   Amount:             {quote.amount} USDC")
            print(f"   Fee:                {quote.fee} USDC ({quote.fee_percentage:.4%})")
            print(f"   Available Liquidity: {quote.available_liquidity:,.0f} USDC")
            print(f"   Estimated Gas:      {quote.gas_estimate:,}")
            print(f"   Total Cost:         {quote.total_cost} USDC")
        else:
            print("❌ No available providers found")
    
    except Exception as e:
        print(f"⚠️  Note: Live RPC call failed (expected in demo): {e}")
        print("   In production, this would return actual liquidity data")
    
    # Demo: Provider fee comparison
    print("\n💰 Provider Fee Comparison:")
    print(f"{'Provider':<20} {'Fee %':<10} {'Fee on $1000':<15} {'Best For'}")
    print("-" * 70)
    
    fees = {
        'Balancer V3': (0.0, 0.0, 'Large trades, zero cost'),
        'Curve': (0.04, 0.40, 'Stablecoin trades'),
        'Aave V3': (0.09, 0.90, 'Guaranteed liquidity'),
    }
    
    for provider, (fee_pct, fee_usd, use_case) in fees.items():
        print(f"{provider:<20} {fee_pct:.2f}%{' ':<6} ${fee_usd:<13.2f} {use_case}")
    
    # Demo: Simultaneous flash loans
    print("\n🔀 Simultaneous Flash Loan Support:")
    print("   The system can execute up to 3 flash loans simultaneously:")
    print("   - One from Aave V3")
    print("   - One from Curve")
    print("   - One from Balancer V3")
    print("   This enables complex multi-step arbitrage strategies!")
    
    # Demo: Example use cases
    print("\n💡 Example Use Cases:")
    print("   1. Liquidation: Flash loan → liquidate position → repay → profit")
    print("   2. Arbitrage:   Flash loan → buy low → sell high → repay → profit")
    print("   3. Multi-step:  3 flash loans → triangular arb → repay all → profit")
    
    print("\n✅ Demo Complete!")
    print("=" * 60)


async def demo_provider_selection_logic():
    """Demonstrate provider selection algorithm"""
    
    print("\n📊 Provider Selection Algorithm Demo")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'Small Trade ($1,000)',
            'amount': 1000,
            'providers': [
                ('Balancer V3', 0, 'Selected - Zero fee'),
                ('Curve', 0.40, 'Alternative'),
                ('Aave V3', 0.90, 'Alternative'),
            ]
        },
        {
            'name': 'Large Trade ($100,000)',
            'amount': 100000,
            'providers': [
                ('Balancer V3', 0, 'Selected - Zero fee'),
                ('Curve', 40, 'Alternative if Balancer busy'),
                ('Aave V3', 90, 'Fallback'),
            ]
        },
        {
            'name': 'Stablecoin Trade with Low Liquidity',
            'amount': 50000,
            'providers': [
                ('Balancer V3', 0, 'Insufficient liquidity'),
                ('Curve', 20, 'Selected - Good liquidity'),
                ('Aave V3', 45, 'Alternative'),
            ]
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Amount: ${scenario['amount']:,}")
        print("Provider Selection:")
        for provider, fee, reason in scenario['providers']:
            print(f"  {provider:<15} Fee: ${fee:<8.2f} - {reason}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  FLASH LOAN MULTI-PROVIDER SYSTEM DEMONSTRATION")
    print("="*60 + "\n")
    
    # Run main demo
    asyncio.run(demo_flash_loan_system())
    
    # Run selection logic demo
    asyncio.run(demo_provider_selection_logic())
    
    print("\n" + "="*60)
    print("  For production use, configure actual contract addresses")
    print("  in .env file and use secure key management.")
    print("="*60 + "\n")
