"""
Card representation for Bridge Monte Carlo Simulator.

This module defines the basic data structures for cards, suits, hands, and decks
used throughout the bridge simulation.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Suit(Enum):
    """Represents the four suits in a deck of cards."""

    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, suit_str: str) -> "Suit":
        """Create Suit from string representation."""
        try:
            return cls(suit_str.strip().upper())
        except ValueError as exc:
            raise ValueError(f"Unknown suit: {suit_str!r}") from exc


class Rank(Enum):
    """Represents card ranks from 2 to Ace."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self):
        if self.value <= 9:
            return str(self.value)
        return {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]

    @classmethod
    def from_string(cls, rank_str: str) -> "Rank":
        """Create Rank from string representation."""
        value = rank_str.strip().upper()
        rank_map = {
            "T": cls.TEN,
            "J": cls.JACK,
            "Q": cls.QUEEN,
            "K": cls.KING,
            "A": cls.ACE,
        }
        try:
            return cls(int(value)) if value.isdigit() else rank_map[value]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown rank: {rank_str!r}") from exc


@dataclass(frozen=True)
class Card:
    """Represents a single playing card."""

    suit: Suit
    rank: Rank

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank.name})"

    @classmethod
    def from_string(cls, card_str: str) -> "Card":
        """Create Card from string like 'AS' (Ace of Spades) or 'TH' (Ten of Hearts)."""
        value = card_str.strip()
        if len(value) not in (2, 3):
            raise ValueError(f"Invalid card string: {card_str}")
        rank_str, suit_str = value[:-1], value[-1]
        return cls(Suit.from_string(suit_str), Rank.from_string(rank_str))

    def __lt__(self, other):
        """Compare cards by rank for sorting."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank.value < other.rank.value


class Hand:
    """Represents a bridge hand of 13 cards."""

    def __init__(self, cards: List[Card] | None = None):
        self.cards = list(cards) if cards is not None else []
        self._validate_hand()

    def _validate_hand(self):
        """Validate that hand has no duplicates and proper size."""
        if len(self.cards) > 13:
            raise ValueError(
                f"Hand cannot have more than 13 cards, got {len(self.cards)}"
            )

        if len(set(self.cards)) != len(self.cards):
            raise ValueError("Hand cannot contain duplicate cards")

    def add_card(self, card: Card):
        """Add a card to the hand."""
        if len(self.cards) >= 13:
            raise ValueError("Cannot add card to full hand")
        if card in self.cards:
            raise ValueError(f"Card {card} already in hand")
        self.cards.append(card)

    def remove_card(self, card: Card):
        """Remove a card from the hand."""
        if card not in self.cards:
            raise ValueError(f"Card {card} not in hand")
        self.cards.remove(card)

    def get_suit_cards(self, suit: Suit) -> List[Card]:
        """Get all cards of a specific suit."""
        return [card for card in self.cards if card.suit == suit]

    def get_suit_length(self, suit: Suit) -> int:
        """Get the number of cards in a specific suit."""
        return len(self.get_suit_cards(suit))

    def get_suit_distribution(self) -> Dict[Suit, int]:
        """Get the distribution of cards by suit."""
        return {suit: self.get_suit_length(suit) for suit in Suit}

    def has_card(self, card: Card) -> bool:
        """Check if hand contains a specific card."""
        return card in self.cards

    def is_void_in_suit(self, suit: Suit) -> bool:
        """Check if hand has no cards in a specific suit."""
        return self.get_suit_length(suit) == 0

    def get_highest_card_in_suit(self, suit: Suit) -> Optional[Card]:
        """Get the highest card in a specific suit."""
        suit_cards = self.get_suit_cards(suit)
        return max(suit_cards, key=lambda c: c.rank.value) if suit_cards else None

    def sort_by_suit_and_rank(self):
        """Sort cards by suit and rank for display."""
        # Sort by suit order (S, H, D, C) then by rank descending
        suit_order = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]

        def sort_key(card):
            return (suit_order.index(card.suit), -card.rank.value)

        self.cards.sort(key=sort_key)

    def __str__(self):
        """String representation showing cards grouped by suit."""
        result = []
        for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
            suit_cards = self.get_suit_cards(suit)
            if suit_cards:
                cards_str = " ".join(
                    str(card.rank)
                    for card in sorted(suit_cards, key=lambda card: -card.rank.value)
                )
                result.append(f"{suit}: {cards_str}")
        return "\n".join(result)

    def __len__(self):
        return len(self.cards)

    @classmethod
    def from_string(cls, hand_str: str) -> "Hand":
        """Create Hand from string representation like 'AS KS QH JD TC'."""
        if not hand_str.strip():
            return cls([])

        card_strings = hand_str.strip().split()
        cards = [Card.from_string(card_str) for card_str in card_strings]
        return cls(cards)


class Deck:
    """Represents a full deck of 52 cards."""

    def __init__(self):
        self.cards = []
        self._create_full_deck()

    def _create_full_deck(self):
        """Create a full deck of 52 cards."""
        self.cards = []
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]

    def shuffle(self):
        """Shuffle the deck randomly."""
        random.shuffle(self.cards)

    def deal_card(self) -> Card:
        """Deal one card from the deck."""
        if not self.cards:
            raise ValueError("Cannot deal from empty deck")
        return self.cards.pop()

    def deal_hand(self, num_cards: int = 13) -> Hand:
        """Deal a hand of specified number of cards."""
        if not 0 <= num_cards <= len(self.cards):
            raise ValueError(
                f"Cannot deal {num_cards} cards from a {len(self.cards)}-card deck"
            )
        return Hand([self.deal_card() for _ in range(num_cards)])

    def remove_cards(self, cards_to_remove: List[Card]):
        """Remove specific cards from the deck (used when player hand is known)."""
        for card in cards_to_remove:
            if card in self.cards:
                self.cards.remove(card)
            else:
                raise ValueError(f"Card {card} not found in deck")

    def remaining_cards(self) -> int:
        """Get number of cards remaining in deck."""
        return len(self.cards)

    def __len__(self):
        return len(self.cards)


def deal_remaining_cards(known_hand: Hand) -> Dict[str, Hand]:
    """
    Deal the remaining 39 cards to North, East, West given South's known hand.

    Args:
        known_hand: The known hand (typically South's)

    Returns:
        Dictionary with keys 'North', 'East', 'West' containing their dealt hands
    """
    if len(known_hand) != 13:
        raise ValueError("Known hand must contain exactly 13 cards")

    deck = Deck()
    deck.remove_cards(known_hand.cards)
    deck.shuffle()

    return {
        "North": deck.deal_hand(13),
        "East": deck.deal_hand(13),
        "West": deck.deal_hand(13),
    }


def parse_hand_input(hand_input: str) -> Hand:
    """
    Parse various hand input formats and return a Hand object.

    Supported formats:
    - Space-separated cards: "AS KS QH JD TC"
    - Suit-grouped format: "S: A K Q | H: J T | D: 9 8 | C: 7 6 5"
    """
    hand_input = hand_input.strip()
    if not hand_input:
        return Hand()

    if "|" in hand_input or ":" in hand_input:
        cards = []
        suit_groups = hand_input.split("|")
        for group in suit_groups:
            if ":" not in group:
                raise ValueError(f"Invalid suit group: {group.strip()!r}")
            suit_str, ranks_str = group.split(":", 1)
            suit = Suit.from_string(suit_str)
            cards.extend(
                Card(suit, Rank.from_string(rank)) for rank in ranks_str.split()
            )
        return Hand(cards)

    # Handle simple space-separated format
    return Hand.from_string(hand_input)


# Test function to validate implementation
def test_card_system():
    """Test the card system implementation."""
    print("Testing Card System...")

    # Test Card creation
    ace_spades = Card(Suit.SPADES, Rank.ACE)
    print(f"Created card: {ace_spades}")

    # Test Hand creation
    cards = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.JACK),
    ]
    hand = Hand(cards)
    print(f"Hand: {hand}")

    # Test dealing
    known_hand = Hand.from_string("AS KS QH JD TC 9C 8C 7C 6C 5C 4C 3C 2C")
    other_hands = deal_remaining_cards(known_hand)
    print(f"North has {len(other_hands['North'])} cards")
    print(f"East has {len(other_hands['East'])} cards")
    print(f"West has {len(other_hands['West'])} cards")

    print("Card system tests completed!")


if __name__ == "__main__":
    test_card_system()
