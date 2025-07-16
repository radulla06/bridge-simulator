"""
Command-line interface for the Bridge Monte Carlo Simulator.

This module provides a user-friendly CLI for running bridge simulations.
"""

import argparse
import sys
from typing import List, Optional
from pathlib import Path

from .cards import Hand, parse_hand_input
from .contracts import Contract
from .monte_carlo import MonteCarloSimulator, ResultsVisualizer, ResultsExporter


def cli_main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Bridge Monte Carlo Contract Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --hand "AS KS QH JH AD KD QD JD TC 9C 8C 7C 6C" --contract "3NT S"
  %(prog)s --hand "S: A K Q | H: J T | D: A K Q J | C: A K Q" --contract "7NT S" --simulations 5000
  %(prog)s --interactive
        """
    )
    
    # Input arguments
    parser.add_argument('--hand', type=str,
                       help='Your hand (e.g., "AS KS QH JH AD KD QD JD TC 9C 8C 7C 6C")')
    parser.add_argument('--contract', type=str,
                       help='Contract to simulate (e.g., "3NT S", "4H N")')
    parser.add_argument('--position', type=str, default='S',
                       choices=['N', 'S', 'E', 'W'],
                       help='Your position (default: S)')
    parser.add_argument('--vulnerable', action='store_true',
                       help='Whether your side is vulnerable')
    
    # Simulation parameters
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Number of simulations to run (default: 10000)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    # Output options
    parser.add_argument('--save-json', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--save-csv', type=str,
                       help='Save summary to CSV file')
    parser.add_argument('--save-plots', type=str,
                       help='Save plots to directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Don\'t show plots')
    
    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple contracts (e.g., "3NT S" "4H S" "5C S")')
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_mode()
    elif args.compare:
        run_comparison_mode(args)
    elif args.hand and args.contract:
        run_single_simulation(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_single_simulation(args):
    """Run a single contract simulation."""
    try:
        # Parse inputs
        hand = parse_hand_input(args.hand)
        contract = Contract.from_string(args.contract)
        
        print(f"Bridge Monte Carlo Simulator")
        print(f"=" * 50)
        print(f"Hand: {hand}")
        print(f"Contract: {contract}")
        print(f"Position: {args.position}")
        print(f"Vulnerable: {args.vulnerable}")
        print(f"Simulations: {args.simulations:,}")
        print()
        
        # Run simulation
        simulator = MonteCarloSimulator(
            num_simulations=args.simulations,
            verbose=not args.quiet
        )
        
        from .contracts import Position
        position = Position.from_string(args.position)
        
        results = simulator.run_simulation(
            hand, contract, position, args.vulnerable
        )
        
        # Save results if requested
        if args.save_json:
            ResultsExporter.save_to_json(results, args.save_json)
            print(f"Results saved to {args.save_json}")
        
        if args.save_csv:
            ResultsExporter.save_to_csv(results, args.save_csv)
            print(f"Summary saved to {args.save_csv}")
        
        # Show plots
        if not args.no_plots:
            visualizer = ResultsVisualizer()
            
            plot_dir = None
            if args.save_plots:
                plot_dir = Path(args.save_plots)
                plot_dir.mkdir(exist_ok=True)
            
            # Trick distribution plot
            save_path = None
            if plot_dir:
                save_path = plot_dir / "trick_distribution.png"
            visualizer.plot_trick_distribution(results, save_path)
            
            # Score distribution plot
            save_path = None
            if plot_dir:
                save_path = plot_dir / "score_distribution.png"
            visualizer.plot_score_distribution(results, save_path)
            
            if plot_dir:
                print(f"Plots saved to {plot_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_comparison_mode(args):
    """Run comparison of multiple contracts."""
    try:
        # Parse inputs
        hand = parse_hand_input(args.hand)
        contracts = [Contract.from_string(c) for c in args.compare]
        
        print(f"Bridge Monte Carlo Simulator - Contract Comparison")
        print(f"=" * 60)
        print(f"Hand: {hand}")
        print(f"Contracts: {[str(c) for c in contracts]}")
        print(f"Position: {args.position}")
        print(f"Vulnerable: {args.vulnerable}")
        print(f"Simulations per contract: {args.simulations:,}")
        print()
        
        # Run comparisons
        simulator = MonteCarloSimulator(
            num_simulations=args.simulations,
            verbose=not args.quiet
        )
        
        from .contracts import Position
        position = Position.from_string(args.position)
        
        results = simulator.compare_contracts(
            hand, contracts, position, args.vulnerable
        )
        
        # Show comparison plot
        if not args.no_plots:
            visualizer = ResultsVisualizer()
            
            save_path = None
            if args.save_plots:
                plot_dir = Path(args.save_plots)
                plot_dir.mkdir(exist_ok=True)
                save_path = plot_dir / "contract_comparison.png"
            
            visualizer.plot_contract_comparison(results, save_path)
            
            if save_path:
                print(f"Comparison plot saved to {save_path}")
        
        # Save individual results if requested
        if args.save_json or args.save_csv:
            for i, (contract_str, result) in enumerate(results.items()):
                if args.save_json:
                    filename = f"{args.save_json.rsplit('.', 1)[0]}_{i+1}.json"
                    ResultsExporter.save_to_json(result, filename)
                
                if args.save_csv:
                    filename = f"{args.save_csv.rsplit('.', 1)[0]}_{i+1}.csv"
                    ResultsExporter.save_to_csv(result, filename)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_interactive_mode():
    """Run the simulator in interactive mode."""
    print("Bridge Monte Carlo Simulator - Interactive Mode")
    print("=" * 50)
    print()
    
    while True:
        try:
            # Get hand input
            print("Enter your hand:")
            print("Format options:")
            print("  1. Space-separated: AS KS QH JH AD KD QD JD TC 9C 8C 7C 6C")
            print("  2. Suit-grouped: S: A K Q | H: J T | D: A K Q J | C: A K Q")
            print()
            
            hand_input = input("Your hand: ").strip()
            if not hand_input:
                print("Goodbye!")
                break
            
            try:
                hand = parse_hand_input(hand_input)
                print(f"Parsed hand: {hand}")
                print()
            except Exception as e:
                print(f"Error parsing hand: {e}")
                continue
            
            # Get contract
            contract_input = input("Contract (e.g., '3NT S', '4H N'): ").strip()
            if not contract_input:
                continue
            
            try:
                contract = Contract.from_string(contract_input)
                print(f"Contract: {contract}")
            except Exception as e:
                print(f"Error parsing contract: {e}")
                continue
            
            # Get additional parameters
            position_input = input("Your position [S/N/E/W] (default S): ").strip() or 'S'
            vulnerable_input = input("Vulnerable? [y/N]: ").strip().lower()
            simulations_input = input("Number of simulations (default 10000): ").strip()
            
            try:
                from .contracts import Position
                position = Position.from_string(position_input)
                vulnerable = vulnerable_input.startswith('y')
                simulations = int(simulations_input) if simulations_input else 10000
            except Exception as e:
                print(f"Error with parameters: {e}")
                continue
            
            print()
            print("Running simulation...")
            
            # Run simulation
            simulator = MonteCarloSimulator(num_simulations=simulations, verbose=True)
            results = simulator.run_simulation(hand, contract, position, vulnerable)
            
            # Show plots
            show_plots = input("\nShow plots? [Y/n]: ").strip().lower()
            if not show_plots.startswith('n'):
                visualizer = ResultsVisualizer()
                visualizer.plot_trick_distribution(results)
                visualizer.plot_score_distribution(results)
            
            print("\n" + "="*50)
            continue_sim = input("Run another simulation? [Y/n]: ").strip().lower()
            if continue_sim.startswith('n'):
                break
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue


def get_sample_hands() -> List[tuple]:
    """Get some sample hands for testing/demonstration."""
    return [
        ("Strong NT opener", "AS KQ QH JH TH AD KD QD JD TC 9C 8C", "3NT S"),
        ("Weak 2 in Hearts", "7S 6S AH KH QH JH TH 9H 8H AD 7D 6D 5D", "2H S"),
        ("Strong 2C opener", "AS KS QS AH KH AH AD KD QD AC KC QC JC", "2C S"),
        ("Preempt in Spades", "AS KS QS JS TS 9S 8S 7S 6S 5S 3H 2H 4D", "3S S"),
    ]


if __name__ == "__main__":
    main() 