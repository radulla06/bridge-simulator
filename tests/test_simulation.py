import random

import pytest

from bridge_simulator.cards import Hand
from bridge_simulator.contracts import Contract, Position
from bridge_simulator.monte_carlo import MonteCarloSimulator
from bridge_simulator.simulation import BridgeSimulator

HAND = Hand.from_string("AS KS QS JS TS 9S 8S AH KH QH AD KD AC")


def test_single_simulation_plays_thirteen_tricks():
    random.seed(7)
    tricks = BridgeSimulator().simulate_hand(
        HAND, Contract.from_string("4S S"), Position.SOUTH
    )
    assert 0 <= tricks <= 13


def test_small_monte_carlo_run_returns_consistent_results():
    random.seed(7)
    results = MonteCarloSimulator(num_simulations=5).run_simulation(
        HAND, Contract.from_string("4S S")
    )

    assert results.num_simulations == 5
    assert len(results.tricks_distribution) == 5
    assert 0 <= results.success_rate <= 1
    assert results.get_trick_statistics()["max"] <= 13


def test_simulation_count_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        MonteCarloSimulator(num_simulations=0)
