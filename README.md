# Bridge Monte Carlo Contract Simulator 🃏

A sophisticated Python program that simulates thousands of random card distributions to estimate the probability of making bridge contracts. Given your hand and a proposed contract, it uses Monte Carlo methods and basic bridge playing logic to provide statistical analysis of your chances.

## 🎯 Features

- **Monte Carlo Simulation**: Run thousands of simulations with random card distributions
- **Realistic Playing Logic**: Basic but sound bridge playing rules and strategies
- **Statistical Analysis**: Success rates, confidence intervals, expected scores
- **Multiple Interfaces**: Command-line interface and modern web UI
- **Visualization**: Interactive charts and histograms
- **Export Options**: Save results to CSV/JSON formats
- **Contract Comparison**: Compare multiple contracts for the same hand

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd bridge-simulator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Command Line Interface:**
```bash
# Simple simulation
python main.py --hand "AS KS QH JH TH AD KD QD JD TC 9C 8C 7C" --contract "3NT S"

# Interactive mode
python main.py --interactive

# Compare contracts
python main.py --hand "AS KS QH JH TH AD KD QD JD TC 9C 8C 7C" --compare "3NT S" "4H S" "5C S"
```

**Web Interface:**
```bash
# Launch Streamlit web app
python main.py --web
# or
streamlit run bridge_simulator/streamlit_app.py
```

## 📋 Input Formats

### Hand Input
**Space-separated cards:**
```
AS KS QH JH TH AD KD QD JD TC 9C 8C 7C
```

**Suit-grouped format:**
```
S: A K Q | H: J T | D: A K Q J | C: A K Q
```

### Contract Input
- `3NT S` - 3 No Trump by South
- `4H N` - 4 Hearts by North  
- `6C E` - 6 Clubs by East
- `7NT W Doubled` - 7 No Trump by West, Doubled

## 🎲 How It Works

1. **Card Distribution**: The simulator takes your known hand and randomly distributes the remaining 39 cards to the other three players
2. **Playing Logic**: Each hand is played out using realistic bridge rules:
   - Must follow suit when possible
   - Trump when void and beneficial
   - Simple but effective card selection heuristics
3. **Statistical Analysis**: Results from thousands of simulations are aggregated to provide:
   - Success probability with confidence intervals
   - Expected number of tricks
   - Score distributions
   - Overtrick/undertrick rates

## 📊 Example Output

```
============================================================
SIMULATION RESULTS: 3NT by S
============================================================
Simulations run: 10,000
Known hand: S

CONTRACT SUCCESS:
  Made rate: 67.3%
  Failed rate: 32.7%
  95% Confidence: 66.4% - 68.2%

TRICK STATISTICS:
  Expected tricks: 8.8
  Median tricks: 9.0
  Standard deviation: 1.2
  Range: 5 - 13
  Quartiles: 8.0 - 10.0

ADDITIONAL STATS:
  Overtrick rate: 23.1%
  Undertrick rate: 32.7%
  Expected score: 289
```

## 🛠️ Advanced Usage

### Command Line Options

```bash
python main.py --help
```

**Key options:**
- `--simulations N`: Number of simulations (default: 10,000)
- `--vulnerable`: Set vulnerability
- `--position P`: Your position (N/S/E/W)
- `--save-json FILE`: Export results to JSON
- `--save-csv FILE`: Export summary to CSV
- `--save-plots DIR`: Save visualizations
- `--quiet`: Suppress progress output

### Programming Interface

```python
from bridge_simulator.cards import Hand
from bridge_simulator.contracts import Contract
from bridge_simulator.monte_carlo import MonteCarloSimulator

# Create hand and contract
hand = Hand.from_string("AS KS QH JH TH AD KD QD JD TC 9C 8C 7C")
contract = Contract.from_string("3NT S")

# Run simulation
simulator = MonteCarloSimulator(num_simulations=10000)
results = simulator.run_simulation(hand, contract)

print(f"Success rate: {results.made_rate:.1f}%")
print(f"Expected tricks: {results.expected_tricks:.1f}")
```

## 🧪 Testing

```bash
# Run built-in tests
python -m bridge_simulator.cards
python -m bridge_simulator.contracts
python -m bridge_simulator.simulation
python -m bridge_simulator.monte_carlo

# Run with pytest (if installed)
pytest
```

## 📈 Visualization Examples

The simulator provides several types of visualizations:

1. **Trick Distribution Histogram**: Shows the frequency of different trick counts
2. **Score Distribution**: Displays the distribution of bridge scores
3. **Contract Comparison**: Side-by-side comparison of multiple contracts

## 🎯 Playing Logic

The simulator implements simplified but realistic bridge playing rules:

### Opening Leads
- Against NT: Lead from longest suit, prefer majors
- Against suits: Lead from longest non-trump suit

### Following Suit
- Must follow suit when possible
- Play high when partner hasn't played yet
- Try to win with lowest winning card
- Play low when can't win

### Trumping
- Trump when void and beneficial
- Don't trump partner's winners
- Play lowest trump when first to trump

### Discarding
- Discard from longest non-trump suit
- Keep winners and potential winners

## 🔧 Architecture

```
bridge_simulator/
├── cards.py          # Card, Hand, Deck classes
├── contracts.py      # Contract and Position classes  
├── simulation.py     # Single hand simulation engine
├── monte_carlo.py    # Monte Carlo aggregation & statistics
├── cli.py           # Command-line interface
└── streamlit_app.py # Web interface
```

## 📚 Dependencies

**Core requirements:**
- Python 3.8+
- numpy
- matplotlib
- tqdm

**Web interface (optional):**
- streamlit
- pandas  
- plotly

## 📝 Resume Highlights

This project demonstrates:

- **Monte Carlo Methods**: Statistical simulation with large sample sizes
- **Object-Oriented Design**: Clean class hierarchy for cards, hands, contracts
- **Data Analysis**: Statistical calculations, confidence intervals, distributions
- **Visualization**: Multiple plotting libraries (matplotlib, plotly)
- **User Interfaces**: Both CLI and modern web interfaces
- **Software Engineering**: Modular design, error handling, documentation

## 🏆 Potential Enhancements

- **Advanced Playing Logic**: More sophisticated card play algorithms
- **Bidding Constraints**: Incorporate bidding information to constrain distributions
- **Machine Learning**: Train models on expert play data
- **Parallelization**: Faster simulations using multiprocessing
- **Database Integration**: Store and analyze historical results
- **Tournament Analysis**: Analyze different scoring systems (IMP, matchpoints)
