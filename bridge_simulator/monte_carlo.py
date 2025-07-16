"""
Monte Carlo aggregation for bridge contract simulation.

This module runs multiple simulations and provides statistical analysis
of contract success rates and trick distributions.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import csv
from pathlib import Path

from .cards import Hand
from .contracts import Contract, Position, calculate_contract_score
from .simulation import BridgeSimulator


@dataclass
class SimulationResults:
    """Container for simulation results and statistics."""
    contract: Contract
    known_hand: Hand
    position: Position
    num_simulations: int
    tricks_distribution: List[int]
    success_rate: float
    expected_tricks: float
    expected_score: float
    confidence_interval_95: Tuple[float, float]
    vulnerable: bool = False
    
    @property
    def made_rate(self) -> float:
        """Percentage of simulations where contract was made."""
        return self.success_rate * 100
    
    @property
    def failed_rate(self) -> float:
        """Percentage of simulations where contract failed."""
        return (1 - self.success_rate) * 100
    
    @property
    def overtrick_rate(self) -> float:
        """Percentage of simulations with overtricks."""
        tricks_needed = self.contract.tricks_needed
        overtricks = sum(1 for tricks in self.tricks_distribution if tricks > tricks_needed)
        return (overtricks / self.num_simulations) * 100
    
    @property
    def undertrick_rate(self) -> float:
        """Percentage of simulations with undertricks."""
        tricks_needed = self.contract.tricks_needed
        undertricks = sum(1 for tricks in self.tricks_distribution if tricks < tricks_needed)
        return (undertricks / self.num_simulations) * 100
    
    def get_trick_statistics(self) -> Dict[str, float]:
        """Get detailed trick statistics."""
        tricks_array = np.array(self.tricks_distribution)
        return {
            'mean': float(np.mean(tricks_array)),
            'median': float(np.median(tricks_array)),
            'std': float(np.std(tricks_array)),
            'min': int(np.min(tricks_array)),
            'max': int(np.max(tricks_array)),
            'q25': float(np.percentile(tricks_array, 25)),
            'q75': float(np.percentile(tricks_array, 75))
        }
    
    def get_score_distribution(self) -> List[int]:
        """Calculate score for each simulation result."""
        scores = []
        for tricks in self.tricks_distribution:
            score = calculate_contract_score(self.contract, tricks, self.vulnerable)
            scores.append(score)
        return scores


class MonteCarloSimulator:
    """Runs Monte Carlo simulations for bridge contracts."""
    
    def __init__(self, num_simulations: int = 10000, verbose: bool = False):
        self.num_simulations = num_simulations
        self.verbose = verbose
        self.simulator = BridgeSimulator(verbose=False)  # Individual sims not verbose
    
    def run_simulation(self, known_hand: Hand, contract: Contract, 
                      position: Position = Position.SOUTH, 
                      vulnerable: bool = False) -> SimulationResults:
        """
        Run Monte Carlo simulation for a bridge contract.
        
        Args:
            known_hand: The known hand (typically the user's hand)
            contract: The contract to simulate
            position: Which position holds the known hand
            vulnerable: Whether declaring side is vulnerable
            
        Returns:
            SimulationResults with comprehensive statistics
        """
        if self.verbose:
            print(f"Running {self.num_simulations:,} simulations for {contract}")
            print(f"Known hand ({position.value}): {known_hand}")
            print()
        
        tricks_won = []
        
        # Run simulations with progress bar
        for i in tqdm(range(self.num_simulations), 
                     desc=f"Simulating {contract}", 
                     disable=not self.verbose):
            tricks = self.simulator.simulate_hand(known_hand, contract, position)
            tricks_won.append(tricks)
        
        # Calculate statistics
        results = self._calculate_statistics(
            tricks_won, contract, known_hand, position, vulnerable
        )
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _calculate_statistics(self, tricks_won: List[int], contract: Contract, 
                            known_hand: Hand, position: Position, 
                            vulnerable: bool) -> SimulationResults:
        """Calculate comprehensive statistics from simulation results."""
        tricks_array = np.array(tricks_won)
        tricks_needed = contract.tricks_needed
        
        # Success rate (made contract)
        successes = np.sum(tricks_array >= tricks_needed)
        success_rate = successes / len(tricks_won)
        
        # Expected values
        expected_tricks = float(np.mean(tricks_array))
        
        # Calculate expected score
        scores = [calculate_contract_score(contract, tricks, vulnerable) 
                 for tricks in tricks_won]
        expected_score = float(np.mean(scores))
        
        # Confidence interval for success rate (binomial)
        confidence_interval = self._calculate_confidence_interval(success_rate, len(tricks_won))
        
        return SimulationResults(
            contract=contract,
            known_hand=known_hand,
            position=position,
            num_simulations=len(tricks_won),
            tricks_distribution=tricks_won,
            success_rate=success_rate,
            expected_tricks=expected_tricks,
            expected_score=expected_score,
            confidence_interval_95=confidence_interval,
            vulnerable=vulnerable
        )
    
    def _calculate_confidence_interval(self, success_rate: float, n: int, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for success rate."""
        z_score = 1.96  # 95% confidence
        if confidence == 0.99:
            z_score = 2.576
        elif confidence == 0.90:
            z_score = 1.645
        
        margin_of_error = z_score * np.sqrt((success_rate * (1 - success_rate)) / n)
        lower = max(0, success_rate - margin_of_error)
        upper = min(1, success_rate + margin_of_error)
        
        return (lower, upper)
    
    def _print_results(self, results: SimulationResults):
        """Print simulation results in a formatted way."""
        print(f"\n{'='*60}")
        print(f"SIMULATION RESULTS: {results.contract}")
        print(f"{'='*60}")
        print(f"Simulations run: {results.num_simulations:,}")
        print(f"Known hand: {results.position.value}")
        print()
        
        print(f"CONTRACT SUCCESS:")
        print(f"  Made rate: {results.made_rate:.1f}%")
        print(f"  Failed rate: {results.failed_rate:.1f}%")
        print(f"  95% Confidence: {results.confidence_interval_95[0]*100:.1f}% - {results.confidence_interval_95[1]*100:.1f}%")
        print()
        
        print(f"TRICK STATISTICS:")
        trick_stats = results.get_trick_statistics()
        print(f"  Expected tricks: {results.expected_tricks:.1f}")
        print(f"  Median tricks: {trick_stats['median']:.1f}")
        print(f"  Standard deviation: {trick_stats['std']:.1f}")
        print(f"  Range: {trick_stats['min']} - {trick_stats['max']}")
        print(f"  Quartiles: {trick_stats['q25']:.1f} - {trick_stats['q75']:.1f}")
        print()
        
        print(f"ADDITIONAL STATS:")
        print(f"  Overtrick rate: {results.overtrick_rate:.1f}%")
        print(f"  Undertrick rate: {results.undertrick_rate:.1f}%")
        print(f"  Expected score: {results.expected_score:.0f}")
        print()
    
    def compare_contracts(self, known_hand: Hand, contracts: List[Contract], 
                         position: Position = Position.SOUTH, 
                         vulnerable: bool = False) -> Dict[str, SimulationResults]:
        """Compare multiple contracts for the same hand."""
        results = {}
        
        for contract in contracts:
            print(f"\nSimulating {contract}...")
            result = self.run_simulation(known_hand, contract, position, vulnerable)
            results[str(contract)] = result
        
        # Print comparison summary
        print(f"\n{'='*80}")
        print("CONTRACT COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Contract':<20} {'Success Rate':<12} {'Expected':<10} {'Expected':<12}")
        print(f"{'':20} {'':12} {'Tricks':<10} {'Score':<12}")
        print("-" * 80)
        
        for contract_str, result in results.items():
            print(f"{contract_str:<20} {result.made_rate:>8.1f}%   "
                  f"{result.expected_tricks:>7.1f}    {result.expected_score:>8.0f}")
        
        return results


class ResultsVisualizer:
    """Creates visualizations for simulation results."""
    
    @staticmethod
    def plot_trick_distribution(results: SimulationResults, save_path: Optional[str] = None):
        """Plot histogram of trick distribution."""
        plt.figure(figsize=(10, 6))
        
        tricks_needed = results.contract.tricks_needed
        tricks = results.tricks_distribution
        
        # Create histogram
        plt.hist(tricks, bins=range(min(tricks), max(tricks) + 2), 
                alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical line for contract level
        plt.axvline(tricks_needed, color='red', linestyle='--', linewidth=2, 
                   label=f'Contract ({tricks_needed} tricks)')
        
        # Add mean line
        plt.axvline(results.expected_tricks, color='green', linestyle='-', linewidth=2,
                   label=f'Expected ({results.expected_tricks:.1f} tricks)')
        
        plt.xlabel('Tricks Won by Declarer')
        plt.ylabel('Frequency')
        plt.title(f'Trick Distribution: {results.contract}\n'
                 f'Success Rate: {results.made_rate:.1f}% '
                 f'({results.num_simulations:,} simulations)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_score_distribution(results: SimulationResults, save_path: Optional[str] = None):
        """Plot histogram of score distribution."""
        plt.figure(figsize=(10, 6))
        
        scores = results.get_score_distribution()
        
        # Separate positive and negative scores
        positive_scores = [s for s in scores if s >= 0]
        negative_scores = [s for s in scores if s < 0]
        
        # Plot histograms
        if positive_scores:
            plt.hist(positive_scores, bins=30, alpha=0.7, color='green', 
                    label=f'Made ({len(positive_scores)} hands)', edgecolor='black')
        
        if negative_scores:
            plt.hist(negative_scores, bins=30, alpha=0.7, color='red', 
                    label=f'Failed ({len(negative_scores)} hands)', edgecolor='black')
        
        # Add expected score line
        plt.axvline(results.expected_score, color='blue', linestyle='-', linewidth=2,
                   label=f'Expected Score: {results.expected_score:.0f}')
        
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(f'Score Distribution: {results.contract}\n'
                 f'Expected Score: {results.expected_score:.0f} '
                 f'({results.num_simulations:,} simulations)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_contract_comparison(results_dict: Dict[str, SimulationResults], 
                               save_path: Optional[str] = None):
        """Plot comparison of multiple contracts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        contracts = list(results_dict.keys())
        success_rates = [results_dict[c].made_rate for c in contracts]
        expected_scores = [results_dict[c].expected_score for c in contracts]
        
        # Success rates
        bars1 = ax1.bar(contracts, success_rates, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Contract Success Rates')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Expected scores
        colors = ['green' if score >= 0 else 'red' for score in expected_scores]
        bars2 = ax2.bar(contracts, expected_scores, color=colors, edgecolor='black')
        ax2.set_ylabel('Expected Score')
        ax2.set_title('Expected Scores')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, score in zip(bars2, expected_scores):
            y_pos = bar.get_height() + (10 if score >= 0 else -30)
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{score:.0f}', ha='center', va='bottom' if score >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ResultsExporter:
    """Export simulation results to various formats."""
    
    @staticmethod
    def save_to_json(results: SimulationResults, file_path: str):
        """Save results to JSON file."""
        # Convert results to serializable format
        data = {
            'contract': str(results.contract),
            'position': results.position.value,
            'num_simulations': results.num_simulations,
            'success_rate': results.success_rate,
            'expected_tricks': results.expected_tricks,
            'expected_score': results.expected_score,
            'confidence_interval_95': results.confidence_interval_95,
            'vulnerable': results.vulnerable,
            'trick_statistics': results.get_trick_statistics(),
            'tricks_distribution': results.tricks_distribution[:1000]  # Limit size
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def save_to_csv(results: SimulationResults, file_path: str):
        """Save summary statistics to CSV file."""
        stats = results.get_trick_statistics()
        
        data = {
            'contract': str(results.contract),
            'position': results.position.value,
            'num_simulations': results.num_simulations,
            'success_rate': results.success_rate,
            'made_rate_percent': results.made_rate,
            'expected_tricks': results.expected_tricks,
            'expected_score': results.expected_score,
            'ci_lower': results.confidence_interval_95[0],
            'ci_upper': results.confidence_interval_95[1],
            'vulnerable': results.vulnerable,
            'min_tricks': stats['min'],
            'max_tricks': stats['max'],
            'median_tricks': stats['median'],
            'std_tricks': stats['std'],
            'q25_tricks': stats['q25'],
            'q75_tricks': stats['q75']
        }
        
        # Write to CSV
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)


# Test function
def test_monte_carlo():
    """Test the Monte Carlo simulation system."""
    print("Testing Monte Carlo System...")
    
    # Create test data
    from .cards import Hand
    from .contracts import Contract
    
    # A reasonable hand for 3NT
    test_hand = Hand.from_string("AK QS QH JH TH AD KD QD JD TC 9C 8C 7C")
    contract = Contract.from_string("3NT S")
    
    print(f"Test hand: {test_hand}")
    print(f"Contract: {contract}")
    
    # Run Monte Carlo simulation (small number for testing)
    simulator = MonteCarloSimulator(num_simulations=1000, verbose=True)
    results = simulator.run_simulation(test_hand, contract)
    
    # Test visualization
    visualizer = ResultsVisualizer()
    visualizer.plot_trick_distribution(results)
    
    print("Monte Carlo tests completed!")


if __name__ == "__main__":
    test_monte_carlo() 