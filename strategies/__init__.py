"""
Trading Strategies

This package contains different trading strategy implementations:
- single_pair_strategy: Original single-pair trading logic
- multi_pair_strategy: 4-pair diversified portfolio strategy
"""

from .single_pair_strategy import SinglePairStrategy
from .multi_pair_strategy import MultiPairStrategy

__all__ = [
    'SinglePairStrategy',
    'MultiPairStrategy'
]
