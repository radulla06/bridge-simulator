import random

import pytest

from bridge_simulator.cards import Card, Deck, Hand, parse_hand_input


def test_card_parser_accepts_ten_in_both_notations():
    assert Card.from_string("TS") == Card.from_string("10s")


def test_hand_rejects_duplicate_cards():
    with pytest.raises(ValueError, match="duplicate"):
        Hand.from_string("AS AS")


def test_hand_string_does_not_reorder_cards():
    hand = Hand.from_string("2C AS KH")
    original_order = hand.cards.copy()

    assert str(hand) == "S: A\nH: K\nC: 2"
    assert hand.cards == original_order


def test_grouped_hand_parser_requires_valid_groups():
    with pytest.raises(ValueError, match="Invalid suit group"):
        parse_hand_input("S: A K | broken")


def test_deck_deals_each_card_once():
    random.seed(42)
    deck = Deck()
    deck.shuffle()
    cards = [deck.deal_card() for _ in range(52)]

    assert len(set(cards)) == 52
    assert len(deck) == 0
