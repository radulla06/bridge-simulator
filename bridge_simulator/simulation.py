"""
Bridge simulation engine for playing out hands.

This module implements the core simulation logic for playing bridge hands
with simplified but realistic playing rules.
"""

from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import random
from .cards import Card, Hand, Suit, Rank, deal_remaining_cards
from .contracts import Contract, Position


class Trick(NamedTuple):
    """Represents a single trick in bridge."""
    leader: Position
    cards: Dict[Position, Card]
    winner: Position
    trump_suit: Optional[Suit]


@dataclass
class GameState:
    """Represents the current state of a bridge game."""
    hands: Dict[Position, Hand]
    contract: Contract
    tricks: List[Trick]
    current_leader: Position
    cards_played: List[Card]
    
    @property
    def declarer_tricks_won(self) -> int:
        """Count tricks won by the declaring partnership."""
        declarer_partnership = [self.contract.declarer, self.contract.declarer.get_partner()]
        return sum(1 for trick in self.tricks if trick.winner in declarer_partnership)
    
    @property
    def defender_tricks_won(self) -> int:
        """Count tricks won by the defending partnership."""
        return len(self.tricks) - self.declarer_tricks_won


class BridgeSimulator:
    """Simulates bridge hands with basic playing logic."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def simulate_hand(self, known_hand: Hand, contract: Contract, position: Position = Position.SOUTH) -> int:
        """
        Simulate a single bridge hand and return tricks won by declarer.
        
        Args:
            known_hand: The known hand (typically the user's hand)
            contract: The contract being played
            position: Which position holds the known hand
            
        Returns:
            Number of tricks won by declaring partnership
        """
        # Deal the remaining cards
        other_hands = deal_remaining_cards(known_hand)
        
        # Create full hand distribution (create a copy of the known hand)
        hands = {position: Hand(known_hand.cards.copy())}
        positions = [pos for pos in Position if pos != position]
        other_positions = ['North', 'East', 'West']
        
        for i, pos in enumerate(positions):
            hands[pos] = other_hands[other_positions[i]]
        
        # Create game state
        game_state = GameState(
            hands=hands,
            contract=contract,
            tricks=[],
            current_leader=contract.declarer.get_left_hand_opponent(),  # Left of declarer leads
            cards_played=[]
        )
        
        if self.verbose:
            print(f"Simulating {contract}")
            print(f"Declarer: {contract.declarer}")
            self._print_hands(hands)
        
        # Play all 13 tricks
        for trick_num in range(13):
            self._play_trick(game_state, trick_num)
        
        if self.verbose:
            print(f"Declarer won {game_state.declarer_tricks_won} tricks")
        
        return game_state.declarer_tricks_won
    
    def _play_trick(self, game_state: GameState, trick_num: int):
        """Play a single trick."""
        trick_cards = {}
        leader = game_state.current_leader
        current_player = leader
        
        # Play 4 cards in clockwise order
        for i in range(4):
            card = self._choose_card_to_play(game_state, current_player, trick_cards, i == 0)
            trick_cards[current_player] = card
            
            # Remove card from hand
            game_state.hands[current_player].remove_card(card)
            game_state.cards_played.append(card)
            
            if self.verbose:
                print(f"  {current_player.value} plays {card}")
            
            # Move to next player clockwise
            current_player = current_player.get_left_hand_opponent()
        
        # Determine trick winner
        winner = self._determine_trick_winner(trick_cards, leader, game_state.contract.trump_suit)
        
        # Create trick record
        trick = Trick(leader, trick_cards, winner, game_state.contract.trump_suit)
        game_state.tricks.append(trick)
        
        # Winner leads next trick
        game_state.current_leader = winner
        
        if self.verbose:
            print(f"  {winner.value} wins the trick")
            print()
    
    def _choose_card_to_play(self, game_state: GameState, player: Position, 
                           trick_cards: Dict[Position, Card], is_leader: bool) -> Card:
        """
        Choose which card to play based on simple heuristics.
        
        This implements a basic strategy:
        - Must follow suit if possible
        - If can't follow suit, trump if beneficial and possible
        - Otherwise, discard lowest card from longest suit
        """
        hand = game_state.hands[player]
        trump_suit = game_state.contract.trump_suit
        
        if is_leader:
            return self._choose_opening_lead(hand, game_state.contract, player)
        
        # Get the suit led
        leader_card = list(trick_cards.values())[0]
        suit_led = leader_card.suit
        
        # Get cards that can follow suit
        following_cards = hand.get_suit_cards(suit_led)
        
        if following_cards:
            # Must follow suit
            return self._choose_from_suit(following_cards, trick_cards, trump_suit, 
                                        player, game_state.contract)
        
        # Can't follow suit - trump or discard
        if trump_suit and not hand.is_void_in_suit(trump_suit):
            # Consider trumping
            if self._should_trump(trick_cards, trump_suit, player, game_state.contract):
                trump_cards = hand.get_suit_cards(trump_suit)
                return self._choose_trump_card(trump_cards, trick_cards)
        
        # Discard
        return self._choose_discard(hand, trump_suit)
    
    def _choose_opening_lead(self, hand: Hand, contract: Contract, leader: Position) -> Card:
        """Choose opening lead based on simple heuristics."""
        trump_suit = contract.trump_suit
        
        # Against no trump: lead longest suit, top of sequence if possible
        if contract.is_no_trump:
            return self._lead_against_notrump(hand)
        
        # Against suit contracts: lead trumps if appropriate, otherwise longest suit
        return self._lead_against_suit_contract(hand, trump_suit, leader, contract)
    
    def _lead_against_notrump(self, hand: Hand) -> Card:
        """Lead against no trump contracts."""
        # Find longest suit
        longest_length = 0
        longest_suits = []
        
        for suit in Suit:
            length = hand.get_suit_length(suit)
            if length > longest_length:
                longest_length = length
                longest_suits = [suit]
            elif length == longest_length:
                longest_suits.append(suit)
        
        # Choose one of the longest suits (prefer majors)
        chosen_suit = longest_suits[0]
        for suit in longest_suits:
            if suit in [Suit.SPADES, Suit.HEARTS]:
                chosen_suit = suit
                break
        
        # Lead top card from chosen suit
        suit_cards = hand.get_suit_cards(chosen_suit)
        return max(suit_cards, key=lambda c: c.rank.value)
    
    def _lead_against_suit_contract(self, hand: Hand, trump_suit: Optional[Suit], 
                                  leader: Position, contract: Contract) -> Card:
        """Lead against suit contracts."""
        # Don't lead trumps unless very strong in trumps
        non_trump_suits = [suit for suit in Suit if suit != trump_suit]
        
        # Find best non-trump suit to lead
        best_suit = None
        best_length = 0
        
        for suit in non_trump_suits:
            length = hand.get_suit_length(suit)
            if length > best_length:
                best_length = length
                best_suit = suit
        
        if best_suit and best_length > 0:
            suit_cards = hand.get_suit_cards(best_suit)
            # Lead top card
            return max(suit_cards, key=lambda c: c.rank.value)
        
        # If only trumps, lead low trump
        if trump_suit:
            trump_cards = hand.get_suit_cards(trump_suit)
            if trump_cards:
                return min(trump_cards, key=lambda c: c.rank.value)
        
        # Fallback: any card
        return hand.cards[0]
    
    def _choose_from_suit(self, cards: List[Card], trick_cards: Dict[Position, Card], 
                         trump_suit: Optional[Suit], player: Position, contract: Contract) -> Card:
        """Choose which card to play when following suit."""
        if len(trick_cards) == 1:
            # Second to play - play high if partner hasn't played yet
            return max(cards, key=lambda c: c.rank.value)
        
        # Look at what's been played
        highest_card = max(trick_cards.values(), key=lambda c: c.rank.value if c.suit == cards[0].suit else 0)
        
        # Try to win if possible and beneficial
        winning_cards = [c for c in cards if c.rank.value > highest_card.rank.value]
        
        if winning_cards:
            # Can win - choose lowest winning card
            return min(winning_cards, key=lambda c: c.rank.value)
        else:
            # Can't win - play lowest card
            return min(cards, key=lambda c: c.rank.value)
    
    def _should_trump(self, trick_cards: Dict[Position, Card], trump_suit: Suit, 
                     player: Position, contract: Contract) -> bool:
        """Decide whether to trump when void in suit led."""
        # Simple heuristic: trump if no one else has trumped yet
        for card in trick_cards.values():
            if card.suit == trump_suit:
                return False  # Someone already trumped
        
        # Trump if partner isn't winning
        cards_so_far = list(trick_cards.values())
        if len(cards_so_far) > 0:
            suit_led = cards_so_far[0].suit
            highest_card = max(cards_so_far, key=lambda c: c.rank.value if c.suit == suit_led else 0)
            
            # Check if partner is winning
            partner = player.get_partner()
            if partner in trick_cards and trick_cards[partner] == highest_card:
                return False  # Partner is winning, don't trump
        
        return True
    
    def _choose_trump_card(self, trump_cards: List[Card], trick_cards: Dict[Position, Card]) -> Card:
        """Choose which trump to play."""
        # Look for trumps already played
        trumps_played = [card for card in trick_cards.values() if card.suit == trump_cards[0].suit]
        
        if trumps_played:
            # Need to play higher trump
            highest_trump = max(trumps_played, key=lambda c: c.rank.value)
            higher_trumps = [c for c in trump_cards if c.rank.value > highest_trump.rank.value]
            
            if higher_trumps:
                return min(higher_trumps, key=lambda c: c.rank.value)  # Lowest winning trump
            else:
                return min(trump_cards, key=lambda c: c.rank.value)  # Can't win, play low
        else:
            # First trump - play low
            return min(trump_cards, key=lambda c: c.rank.value)
    
    def _choose_discard(self, hand: Hand, trump_suit: Optional[Suit]) -> Card:
        """Choose a card to discard when can't follow suit."""
        # Discard from longest non-trump suit
        best_suit = None
        best_length = 0
        
        for suit in Suit:
            if suit != trump_suit:
                length = hand.get_suit_length(suit)
                if length > best_length:
                    best_length = length
                    best_suit = suit
        
        if best_suit and best_length > 0:
            suit_cards = hand.get_suit_cards(best_suit)
            return min(suit_cards, key=lambda c: c.rank.value)  # Discard lowest
        
        # Fallback: any non-trump card
        for card in hand.cards:
            if card.suit != trump_suit:
                return card
        
        # Only trumps left
        return min(hand.cards, key=lambda c: c.rank.value)
    
    def _determine_trick_winner(self, trick_cards: Dict[Position, Card], 
                              leader: Position, trump_suit: Optional[Suit]) -> Position:
        """Determine who wins the trick."""
        cards_played = list(trick_cards.values())
        suit_led = cards_played[0].suit
        
        # Check for trumps (if trump suit exists and someone trumped)
        if trump_suit:
            trump_cards = [(pos, card) for pos, card in trick_cards.items() 
                          if card.suit == trump_suit]
            
            if trump_cards and trump_suit != suit_led:
                # Someone trumped - highest trump wins
                winner_pos, winner_card = max(trump_cards, key=lambda x: x[1].rank.value)
                return winner_pos
        
        # No trumps or trump was led - highest card of suit led wins
        suit_led_cards = [(pos, card) for pos, card in trick_cards.items() 
                         if card.suit == suit_led]
        
        if suit_led_cards:
            winner_pos, winner_card = max(suit_led_cards, key=lambda x: x[1].rank.value)
            return winner_pos
        
        # Shouldn't happen in valid bridge, but fallback to leader
        return leader
    
    def _print_hands(self, hands: Dict[Position, Hand]):
        """Print all hands for debugging."""
        print("\nHands:")
        for position in [Position.NORTH, Position.EAST, Position.SOUTH, Position.WEST]:
            print(f"{position.value}:")
            print(f"  {hands[position]}")


def simulate_contract_once(known_hand: Hand, contract: Contract, 
                          position: Position = Position.SOUTH, verbose: bool = False) -> int:
    """
    Convenience function to simulate a single hand.
    
    Returns the number of tricks won by the declaring partnership.
    """
    simulator = BridgeSimulator(verbose=verbose)
    return simulator.simulate_hand(known_hand, contract, position)


# Test function
def test_simulation():
    """Test the simulation engine."""
    print("Testing Simulation Engine...")
    
    # Create a test hand
    from .cards import Hand
    from .contracts import Contract
    
    test_hand = Hand.from_string("AS KS QS JS TS AH KH QH JH AD KD QD JD")
    contract = Contract.from_string("7NT S")
    
    print(f"Test hand: {test_hand}")
    print(f"Contract: {contract}")
    
    # Run a single simulation
    tricks_won = simulate_contract_once(test_hand, contract, verbose=True)
    print(f"Tricks won: {tricks_won}")
    print(f"Contract {'made' if tricks_won >= contract.tricks_needed else 'failed'}")
    
    print("Simulation tests completed!")


if __name__ == "__main__":
    test_simulation() 