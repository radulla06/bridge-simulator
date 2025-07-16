"""
Contract representation for Bridge Monte Carlo Simulator.

This module defines contracts, bidding, and game rules for bridge simulation.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union
from .cards import Suit


class ContractSuit(Enum):
    """Represents the trump suit or no trump for a contract."""
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"
    NO_TRUMP = "NT"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, suit_str: str) -> 'ContractSuit':
        """Create ContractSuit from string representation."""
        suit_map = {
            'C': cls.CLUBS,
            'D': cls.DIAMONDS, 
            'H': cls.HEARTS,
            'S': cls.SPADES,
            'NT': cls.NO_TRUMP,
            'N': cls.NO_TRUMP  # Alternative notation
        }
        return suit_map[suit_str.upper()]
    
    def to_suit(self) -> Optional[Suit]:
        """Convert to Suit enum if this is a suit contract."""
        if self == self.NO_TRUMP:
            return None
        return Suit(self.value)


class Position(Enum):
    """Represents the four positions in bridge."""
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, pos_str: str) -> 'Position':
        """Create Position from string representation."""
        pos_map = {'N': cls.NORTH, 'E': cls.EAST, 'S': cls.SOUTH, 'W': cls.WEST}
        return pos_map[pos_str.upper()]
    
    def is_ns_partnership(self) -> bool:
        """Check if this position is North-South partnership."""
        return self in [self.NORTH, self.SOUTH]
    
    def is_ew_partnership(self) -> bool:
        """Check if this position is East-West partnership."""
        return self in [self.EAST, self.WEST]
    
    def get_partner(self) -> 'Position':
        """Get the partner position."""
        partners = {
            self.NORTH: self.SOUTH,
            self.SOUTH: self.NORTH,
            self.EAST: self.WEST,
            self.WEST: self.EAST
        }
        return partners[self]
    
    def get_left_hand_opponent(self) -> 'Position':
        """Get the left-hand opponent (next player clockwise)."""
        order = [self.NORTH, self.EAST, self.SOUTH, self.WEST]
        current_index = order.index(self)
        return order[(current_index + 1) % 4]
    
    def get_right_hand_opponent(self) -> 'Position':
        """Get the right-hand opponent (next player counter-clockwise)."""
        order = [self.NORTH, self.EAST, self.SOUTH, self.WEST]
        current_index = order.index(self)
        return order[(current_index - 1) % 4]


@dataclass
class Contract:
    """Represents a bridge contract."""
    level: int  # 1-7
    suit: ContractSuit
    declarer: Position
    doubled: bool = False
    redoubled: bool = False
    
    def __post_init__(self):
        """Validate contract parameters."""
        if not 1 <= self.level <= 7:
            raise ValueError(f"Contract level must be 1-7, got {self.level}")
        
        if self.redoubled and not self.doubled:
            raise ValueError("Cannot be redoubled without being doubled first")
    
    @property
    def tricks_needed(self) -> int:
        """Calculate the number of tricks needed to make this contract."""
        return 6 + self.level
    
    @property
    def trump_suit(self) -> Optional[Suit]:
        """Get the trump suit, or None for no trump."""
        return self.suit.to_suit()
    
    @property
    def is_no_trump(self) -> bool:
        """Check if this is a no trump contract."""
        return self.suit == ContractSuit.NO_TRUMP
    
    @property
    def is_major_suit(self) -> bool:
        """Check if this is a major suit contract (Hearts or Spades)."""
        return self.suit in [ContractSuit.HEARTS, ContractSuit.SPADES]
    
    @property
    def is_minor_suit(self) -> bool:
        """Check if this is a minor suit contract (Clubs or Diamonds)."""
        return self.suit in [ContractSuit.CLUBS, ContractSuit.DIAMONDS]
    
    @property
    def partnership(self) -> str:
        """Get the declaring partnership."""
        return "NS" if self.declarer.is_ns_partnership() else "EW"
    
    def __str__(self):
        """String representation of the contract."""
        base = f"{self.level}{self.suit} by {self.declarer.value}"
        if self.redoubled:
            return f"{base} Redoubled"
        elif self.doubled:
            return f"{base} Doubled"
        return base
    
    @classmethod
    def from_string(cls, contract_str: str) -> 'Contract':
        """
        Create Contract from string representation.
        
        Examples:
        - "3NT by S"
        - "4H by N Doubled"
        - "7C by E Redoubled"
        - "1NT S" (short form)
        """
        parts = contract_str.strip().split()
        
        # Parse the basic contract (e.g., "3NT" or "4H")
        if len(parts[0]) < 2:
            raise ValueError(f"Invalid contract format: {contract_str}")
        
        level_str = parts[0][0]
        suit_str = parts[0][1:]
        
        level = int(level_str)
        suit = ContractSuit.from_string(suit_str)
        
        # Parse declarer
        if len(parts) >= 2:
            if parts[1].lower() == "by" and len(parts) >= 3:
                declarer = Position.from_string(parts[2])
                remaining_parts = parts[3:]
            else:
                declarer = Position.from_string(parts[1])
                remaining_parts = parts[2:]
        else:
            raise ValueError(f"Declarer position not specified: {contract_str}")
        
        # Parse doubling
        doubled = False
        redoubled = False
        
        for part in remaining_parts:
            if part.lower() in ["doubled", "dbl", "x"]:
                doubled = True
            elif part.lower() in ["redoubled", "rdbl", "xx"]:
                doubled = True
                redoubled = True
        
        return cls(level, suit, declarer, doubled, redoubled)


def calculate_contract_score(contract: Contract, tricks_taken: int, vulnerable: bool = False) -> int:
    """
    Calculate the score for a contract given the number of tricks taken.
    
    Args:
        contract: The contract being played
        tricks_taken: Number of tricks taken by declaring side
        vulnerable: Whether the declaring side is vulnerable
    
    Returns:
        The score (positive for made, negative for failed)
    """
    tricks_needed = contract.tricks_needed
    overtricks = tricks_taken - tricks_needed
    
    if overtricks >= 0:
        # Contract made
        return _calculate_made_score(contract, overtricks, vulnerable)
    else:
        # Contract failed
        undertricks = -overtricks
        return -_calculate_penalty_score(contract, undertricks, vulnerable)


def _calculate_made_score(contract: Contract, overtricks: int, vulnerable: bool) -> int:
    """Calculate score for a made contract."""
    # Basic scores
    if contract.is_no_trump:
        basic_score = 40 + (contract.level - 1) * 30
    elif contract.is_major_suit:
        basic_score = contract.level * 30
    else:  # Minor suit
        basic_score = contract.level * 20
    
    # Apply doubling
    if contract.redoubled:
        basic_score *= 4
    elif contract.doubled:
        basic_score *= 2
    
    # Game bonus
    game_bonus = 0
    if basic_score >= 100:  # Game
        game_bonus = 500 if vulnerable else 300
    else:  # Part game
        game_bonus = 50
    
    # Slam bonuses
    slam_bonus = 0
    if contract.level == 7:  # Grand slam
        slam_bonus = 1500 if vulnerable else 1000
    elif contract.level == 6:  # Small slam
        slam_bonus = 750 if vulnerable else 500
    
    # Doubling bonus
    double_bonus = 0
    if contract.doubled:
        double_bonus = 50
    elif contract.redoubled:
        double_bonus = 100
    
    # Overtrick scoring
    overtrick_score = 0
    if overtricks > 0:
        if contract.doubled or contract.redoubled:
            # Doubled overtricks
            multiplier = 2 if contract.redoubled else 1
            if vulnerable:
                overtrick_score = overtricks * 200 * multiplier
            else:
                overtrick_score = overtricks * 100 * multiplier
        else:
            # Normal overtricks
            if contract.is_no_trump:
                overtrick_score = overtricks * 30
            elif contract.is_major_suit:
                overtrick_score = overtricks * 30
            else:  # Minor suit
                overtrick_score = overtricks * 20
    
    return basic_score + game_bonus + slam_bonus + double_bonus + overtrick_score


def _calculate_penalty_score(contract: Contract, undertricks: int, vulnerable: bool) -> int:
    """Calculate penalty score for a failed contract."""
    if not (contract.doubled or contract.redoubled):
        # Undoubled penalties
        return undertricks * (100 if vulnerable else 50)
    
    # Doubled penalties
    multiplier = 2 if contract.redoubled else 1
    penalty = 0
    
    for i in range(undertricks):
        if i == 0:  # First undertrick
            penalty += (200 if vulnerable else 100) * multiplier
        elif i <= 2:  # Second and third undertricks
            penalty += (300 if vulnerable else 200) * multiplier
        else:  # Fourth and subsequent
            penalty += (300 if vulnerable else 300) * multiplier
    
    return penalty


# Test function
def test_contracts():
    """Test the contract system."""
    print("Testing Contract System...")
    
    # Test contract creation
    contract = Contract.from_string("3NT by S")
    print(f"Contract: {contract}")
    print(f"Tricks needed: {contract.tricks_needed}")
    print(f"Is no trump: {contract.is_no_trump}")
    
    # Test scoring
    score = calculate_contract_score(contract, 9, vulnerable=False)
    print(f"Score for making exactly: {score}")
    
    score = calculate_contract_score(contract, 10, vulnerable=False)
    print(f"Score for making +1: {score}")
    
    score = calculate_contract_score(contract, 8, vulnerable=False)
    print(f"Score for going down 1: {score}")
    
    print("Contract system tests completed!")


if __name__ == "__main__":
    test_contracts() 