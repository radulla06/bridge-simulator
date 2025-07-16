"""
Streamlit web interface for the Bridge Monte Carlo Simulator.

Run with: streamlit run bridge_simulator/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List

try:
    from .cards import Hand, parse_hand_input
    from .contracts import Contract, Position
    from .monte_carlo import MonteCarloSimulator, SimulationResults
except ImportError:
    # Handle relative imports when running directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from bridge_simulator.cards import Hand, parse_hand_input
    from bridge_simulator.contracts import Contract, Position
    from bridge_simulator.monte_carlo import MonteCarloSimulator, SimulationResults


# Page configuration
st.set_page_config(
    page_title="Bridge Monte Carlo Simulator",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-rate {
        font-size:24px !important;
        font-weight: bold;
        color: #28a745;
    }
    .failed-rate {
        font-size:24px !important;
        font-weight: bold;
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    st.title("🃏 Bridge Monte Carlo Contract Simulator")
    st.markdown("Simulate thousands of bridge hands to estimate your contract success probability!")
    
    # Sidebar for input
    st.sidebar.header("🎯 Simulation Parameters")
    
    # Hand input
    st.sidebar.subheader("Your Hand")
    hand_input_method = st.sidebar.radio(
        "Input method:",
        ["Card list", "Suit groups", "Example hands"]
    )
    
    if hand_input_method == "Card list":
        hand_input = st.sidebar.text_area(
            "Enter cards (space-separated):",
            value="AS KS QH JH TH AD KD QD JD TC 9C 8C 7C",
            help="Example: AS KS QH JH TH AD KD QD JD TC 9C 8C 7C"
        )
    elif hand_input_method == "Suit groups":
        hand_input = st.sidebar.text_area(
            "Enter by suits:",
            value="S: A K Q | H: J T | D: A K Q J | C: A K Q",
            help="Example: S: A K Q | H: J T | D: A K Q J | C: A K Q"
        )
    else:  # Example hands
        example_hands = {
            "Strong 1NT opener": "AS KQ QH JH TH AD KD QD JD TC 9C 8C 7C",
            "Weak 2 in Hearts": "7S 6S AH KH QH JH TH 9H 8H AD 7D 6D 5D",
            "Strong 2♣ opener": "AS KS QS AH KH QH AD KD QD AC KC QC JC",
            "3♠ preempt": "AS KS QS JS TS 9S 8S 7S 6S 5S 3H 2H 4D",
            "Slam hand": "AS KS QS JS AH KH QH AD KD QD AC KC QC"
        }
        selected_example = st.sidebar.selectbox("Choose example:", list(example_hands.keys()))
        hand_input = example_hands[selected_example]
        st.sidebar.text_area("Hand:", value=hand_input, disabled=True)
    
    # Contract input
    st.sidebar.subheader("Contract")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        level = st.selectbox("Level:", [1, 2, 3, 4, 5, 6, 7], index=2)
    with col2:
        suit = st.selectbox("Suit:", ["♠", "♥", "♦", "♣", "NT"], index=4)
    
    suit_map = {"♠": "S", "♥": "H", "♦": "D", "♣": "C", "NT": "NT"}
    contract_suit = suit_map[suit]
    
    declarer = st.sidebar.selectbox("Declarer:", ["North", "East", "South", "West"], index=2)
    declarer_map = {"North": "N", "East": "E", "South": "S", "West": "W"}
    declarer_pos = declarer_map[declarer]
    
    vulnerable = st.sidebar.checkbox("Vulnerable", value=False)
    
    # Position of known hand
    your_position = st.sidebar.selectbox(
        "Your position:", 
        ["North", "East", "South", "West"], 
        index=2,
        help="Which position holds the known hand"
    )
    position_map = {"North": "N", "East": "E", "South": "S", "West": "W"}
    your_pos = position_map[your_position]
    
    # Simulation settings
    st.sidebar.subheader("Simulation Settings")
    num_simulations = st.sidebar.slider(
        "Number of simulations:",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="More simulations = more accurate results (but slower)"
    )
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        show_detailed_stats = st.checkbox("Show detailed statistics", value=True)
        show_score_analysis = st.checkbox("Show score analysis", value=True)
        export_results = st.checkbox("Enable result export", value=False)
    
    # Run simulation button
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")
    
    # Main content area
    if run_simulation:
        try:
            # Parse inputs
            hand = parse_hand_input(hand_input)
            contract_str = f"{level}{contract_suit} {declarer_pos}"
            contract = Contract.from_string(contract_str)
            position = Position.from_string(your_pos)
            
            # Display inputs
            st.header("📊 Simulation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Your Hand")
                st.text(str(hand))
            
            with col2:
                st.subheader("Contract")
                st.markdown(f"<p class='medium-font'>{contract}</p>", unsafe_allow_html=True)
                if vulnerable:
                    st.markdown("**Vulnerable** 🔴")
                else:
                    st.markdown("**Not Vulnerable** 🟢")
            
            with col3:
                st.subheader("Simulation")
                st.write(f"**Simulations:** {num_simulations:,}")
                st.write(f"**Your position:** {your_position}")
            
            # Run the simulation
            with st.spinner('Running simulation... This may take a moment.'):
                simulator = MonteCarloSimulator(num_simulations=num_simulations, verbose=False)
                results = simulator.run_simulation(hand, contract, position, vulnerable)
            
            # Display key results
            st.subheader("🎯 Key Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                success_color = "success-rate" if results.made_rate >= 50 else "failed-rate"
                st.markdown(f"<p class='{success_color}'>{results.made_rate:.1f}%</p>", unsafe_allow_html=True)
                st.write("**Success Rate**")
            
            with col2:
                st.markdown(f"<p class='medium-font'>{results.expected_tricks:.1f}</p>", unsafe_allow_html=True)
                st.write("**Expected Tricks**")
            
            with col3:
                score_color = "success-rate" if results.expected_score >= 0 else "failed-rate"
                st.markdown(f"<p class='{score_color}'>{results.expected_score:.0f}</p>", unsafe_allow_html=True)
                st.write("**Expected Score**")
            
            with col4:
                ci_lower, ci_upper = results.confidence_interval_95
                st.markdown(f"<p class='medium-font'>{ci_lower*100:.1f}% - {ci_upper*100:.1f}%</p>", unsafe_allow_html=True)
                st.write("**95% Confidence**")
            
            # Visualizations
            create_visualizations(results)
            
            # Detailed statistics
            if show_detailed_stats:
                show_detailed_statistics(results)
            
            # Score analysis
            if show_score_analysis:
                show_score_analysis_section(results)
            
            # Export options
            if export_results:
                show_export_options(results)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your hand and contract input format.")
    
    else:
        # Show introduction when no simulation is running
        show_introduction()


def create_visualizations(results: SimulationResults):
    """Create interactive visualizations using Plotly."""
    st.subheader("📈 Distribution Analysis")
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Trick Distribution**")
        
        # Trick distribution histogram
        tricks = results.tricks_distribution
        tricks_needed = results.contract.tricks_needed
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=tricks,
            nbinsx=max(tricks) - min(tricks) + 1,
            name="Frequency",
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Contract line
        fig.add_vline(
            x=tricks_needed,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Contract ({tricks_needed} tricks)",
            annotation_position="top right"
        )
        
        # Expected line
        fig.add_vline(
            x=results.expected_tricks,
            line_color="green",
            annotation_text=f"Expected ({results.expected_tricks:.1f})",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=f"Success Rate: {results.made_rate:.1f}%",
            xaxis_title="Tricks Won by Declarer",
            yaxis_title="Frequency",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Score Distribution**")
        
        # Score distribution
        scores = results.get_score_distribution()
        positive_scores = [s for s in scores if s >= 0]
        negative_scores = [s for s in scores if s < 0]
        
        fig = go.Figure()
        
        if positive_scores:
            fig.add_trace(go.Histogram(
                x=positive_scores,
                name=f"Made ({len(positive_scores)} hands)",
                marker_color='green',
                opacity=0.7,
                nbinsx=30
            ))
        
        if negative_scores:
            fig.add_trace(go.Histogram(
                x=negative_scores,
                name=f"Failed ({len(negative_scores)} hands)",
                marker_color='red',
                opacity=0.7,
                nbinsx=30
            ))
        
        # Expected score line
        fig.add_vline(
            x=results.expected_score,
            line_color="blue",
            annotation_text=f"Expected: {results.expected_score:.0f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=f"Expected Score: {results.expected_score:.0f}",
            xaxis_title="Score",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_detailed_statistics(results: SimulationResults):
    """Show detailed statistical analysis."""
    st.subheader("📊 Detailed Statistics")
    
    trick_stats = results.get_trick_statistics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Trick Statistics**")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', '25th Percentile', '75th Percentile'],
            'Value': [
                f"{trick_stats['mean']:.2f}",
                f"{trick_stats['median']:.1f}",
                f"{trick_stats['std']:.2f}",
                f"{trick_stats['min']}",
                f"{trick_stats['max']}",
                f"{trick_stats['q25']:.1f}",
                f"{trick_stats['q75']:.1f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Outcome Rates**")
        outcome_df = pd.DataFrame({
            'Outcome': ['Contract Made', 'Contract Failed', 'Overtricks', 'Undertricks'],
            'Percentage': [
                f"{results.made_rate:.1f}%",
                f"{results.failed_rate:.1f}%",
                f"{results.overtrick_rate:.1f}%",
                f"{results.undertrick_rate:.1f}%"
            ]
        })
        st.dataframe(outcome_df, use_container_width=True, hide_index=True)


def show_score_analysis_section(results: SimulationResults):
    """Show score analysis."""
    st.subheader("💰 Score Analysis")
    
    scores = results.get_score_distribution()
    scores_array = np.array(scores)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Score", f"{np.mean(scores_array):.0f}")
        st.metric("Best Result", f"{np.max(scores_array):.0f}")
    
    with col2:
        st.metric("Median Score", f"{np.median(scores_array):.0f}")
        st.metric("Worst Result", f"{np.min(scores_array):.0f}")
    
    with col3:
        positive_rate = len([s for s in scores if s > 0]) / len(scores) * 100
        st.metric("Positive Score Rate", f"{positive_rate:.1f}%")
        st.metric("Score Std Dev", f"{np.std(scores_array):.0f}")


def show_export_options(results: SimulationResults):
    """Show export options for results."""
    st.subheader("📁 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        trick_stats = results.get_trick_statistics()
        export_data = {
            'Contract': [str(results.contract)],
            'Position': [results.position.value],
            'Simulations': [results.num_simulations],
            'Success_Rate': [results.success_rate],
            'Expected_Tricks': [results.expected_tricks],
            'Expected_Score': [results.expected_score],
            'Made_Rate_Percent': [results.made_rate],
            'CI_Lower': [results.confidence_interval_95[0]],
            'CI_Upper': [results.confidence_interval_95[1]],
            'Min_Tricks': [trick_stats['min']],
            'Max_Tricks': [trick_stats['max']],
            'Median_Tricks': [trick_stats['median']],
            'Std_Tricks': [trick_stats['std']]
        }
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV Summary",
            data=csv,
            file_name=f"bridge_simulation_{results.contract.level}{results.contract.suit.value}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export (sample data only due to size)
        import json
        json_data = {
            'contract': str(results.contract),
            'position': results.position.value,
            'num_simulations': results.num_simulations,
            'success_rate': results.success_rate,
            'expected_tricks': results.expected_tricks,
            'expected_score': results.expected_score,
            'confidence_interval_95': results.confidence_interval_95,
            'trick_statistics': trick_stats,
            'sample_tricks': results.tricks_distribution[:100]  # First 100 results
        }
        
        json_str = json.dumps(json_data, indent=2)
        
        st.download_button(
            label="📄 Download JSON Data",
            data=json_str,
            file_name=f"bridge_simulation_{results.contract.level}{results.contract.suit.value}.json",
            mime="application/json"
        )


def show_introduction():
    """Show introduction and help information."""
    st.header("Welcome to the Bridge Monte Carlo Simulator! 🎴")
    
    st.markdown("""
    This tool simulates thousands of possible card distributions to estimate your chances 
    of making a bridge contract. Simply enter your hand, specify the contract, and let 
    the simulator analyze your prospects!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 How it works")
        st.markdown("""
        1. **Enter your hand** using the sidebar
        2. **Choose the contract** you want to analyze
        3. **Set simulation parameters** (number of runs, vulnerability)
        4. **Click "Run Simulation"** to see your results
        
        The simulator will randomly deal the remaining 39 cards to the other players 
        thousands of times and play out each hand using realistic bridge logic.
        """)
    
    with col2:
        st.subheader("📋 Hand input formats")
        st.markdown("""
        **Card list format:**
        ```
        AS KS QH JH TH AD KD QD JD TC 9C 8C 7C
        ```
        
        **Suit groups format:**
        ```
        S: A K Q | H: J T | D: A K Q J | C: A K Q
        ```
        
        Use standard notation: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
        """)
    
    st.subheader("📊 What you'll get")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Success Analysis**
        - Contract success probability
        - 95% confidence interval
        - Expected number of tricks
        - Overtrick/undertrick rates
        """)
    
    with col2:
        st.markdown("""
        **Score Analysis**
        - Expected score
        - Score distribution
        - Best/worst case scenarios
        - Positive score probability
        """)
    
    with col3:
        st.markdown("""
        **Visualizations**
        - Interactive trick histograms
        - Score distribution plots
        - Statistical summaries
        - Export capabilities
        """)
    
    st.info("💡 **Tip:** Start with the example hands to see how the simulator works!")


if __name__ == "__main__":
    main() 