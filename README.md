# Bridge Monte Carlo Contract Simulator

This project estimates how often a bridge contract will make from one known hand. For each run, it deals the other 39 cards at random, plays all 13 tricks with a simple rule-based player, and summarizes the results.

The play engine is deliberately lightweight. It follows suit, handles trumps, and uses basic lead and card-selection rules, but it is not a double-dummy solver and should not be treated as expert analysis.

## Features

- Run a single contract simulation from the command line or web interface.
- Compare several contracts for the same hand.
- Report make rate, expected tricks, expected score, and a 95% confidence interval.
- Plot trick and score distributions.
- Export a result summary as CSV or JSON.
- Account for vulnerability, doubles, and redoubles when calculating duplicate bridge scores.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Command-line usage

A hand must contain exactly 13 unique cards. Cards use rank followed by suit, such as `AS` for the ace of spades or `TC` for the ten of clubs.

Run a simulation:

```bash
python main.py \
  --hand "AS KS QH JH TH AD KD QD JD TC 9C 8C 7C" \
  --contract "3NT S" \
  --simulations 10000
```

By default, the command opens result plots. Add `--no-plots` when running without a graphical display:

```bash
python main.py \
  --hand "AS KS QH JH TH AD KD QD JD TC 9C 8C 7C" \
  --contract "3NT S" \
  --simulations 10000 \
  --no-plots
```

Compare contracts by replacing `--contract` with `--compare`:

```bash
python main.py \
  --hand "AS KS QH JH TH AD KD QD JD TC 9C 8C 7C" \
  --compare "3NT S" "4H S" "5C S" \
  --no-plots
```

For prompts instead of command-line arguments, use:

```bash
python main.py --interactive
```

Run `python main.py --help` for all options, including vulnerability, known-hand position, and result export paths.

## Input formats

Hands may be entered as a card list:

```text
AS KS QH JH TH AD KD QD JD TC 9C 8C 7C
```

They may also be grouped by suit:

```text
S: A K | H: Q J T | D: A K Q J | C: T 9 8 7
```

Use `A`, `K`, `Q`, `J`, `T` (or `10`), and `9` through `2` for ranks. Suits are `S`, `H`, `D`, and `C`.

A contract consists of its level and strain followed by the declarer:

```text
3NT S
4H N
6C E Doubled
7NT W Redoubled
```

The words `by`, `Doubled`, and `Redoubled` are optional syntax where applicable. Short forms such as `x` and `xx` are also accepted.

## Web interface

Launch the Streamlit app through the project entry point:

```bash
python main.py --web
```

You can also run it directly:

```bash
streamlit run bridge_simulator/streamlit_app.py
```

The web interface provides example hands, contract controls, charts, detailed statistics, and downloadable summaries.

## Using the package

```python
from bridge_simulator.cards import Hand
from bridge_simulator.contracts import Contract
from bridge_simulator.monte_carlo import MonteCarloSimulator

hand = Hand.from_string("AS KS QH JH TH AD KD QD JD TC 9C 8C 7C")
contract = Contract.from_string("3NT S")

simulator = MonteCarloSimulator(num_simulations=10_000)
results = simulator.run_simulation(hand, contract)

print(f"Make rate: {results.made_rate:.1f}%")
print(f"Expected tricks: {results.expected_tricks:.1f}")
print(f"Expected score: {results.expected_score:.0f}")
```

## Tests and checks

Run the test suite with:

```bash
pytest -q
```

The repository also includes configuration for Black and Ruff:

```bash
black --check .
ruff check .
```
