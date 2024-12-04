import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import streamlit as st

# Define the simulation function
def simulate_betting_market(num_bettors=100, variance_ability=0.1, num_rounds=1000, bet_fraction=0.1):
    abilities = np.clip(np.random.normal(0.5, variance_ability, num_bettors), 0, 1)
    wealth = np.ones(num_bettors)

    expected_values_over_time = []

    for _ in range(num_rounds):
        bets = bet_fraction * wealth
        weighted_ability = np.sum(abilities * bets) / np.sum(bets)
        expected_values_over_time.append(weighted_ability)

        outcomes = np.random.rand(num_bettors) < abilities
        wealth += bets * (2 * outcomes - 1)
        wealth = np.maximum(wealth, 0)

    results = pd.DataFrame({
        "Ability": abilities,
        "End Wealth": wealth
    })

    results["Wealth Percentile"] = results["End Wealth"].rank(pct=True) * 100
    results["Ability Percentile"] = results["Ability"].rank(pct=True) * 100
    results["Ability^2"] = results["Ability"] ** 2
    results["Ability_exp"] = np.exp(results["Ability"])

    X_wealth = results[["Ability", "Ability^2", "Ability_exp"]]
    X_wealth = sm.add_constant(X_wealth)
    y_wealth = results["End Wealth"]
    model_wealth = sm.OLS(y_wealth, X_wealth).fit()

    y_percentile = results["Wealth Percentile"]
    model_percentile = sm.OLS(y_percentile, X_wealth).fit()

    return results, model_wealth, model_percentile, expected_values_over_time

# Streamlit application
st.title("Betting Market Simulation")
st.sidebar.header("Simulation Parameters")

# Sidebar inputs
num_bettors = st.sidebar.slider("Number of Bettors", 10, 500, 97, step=1)
variance_ability = st.sidebar.slider("Variance of Ability", 0.01, 0.5, 0.1, step=0.01)
num_rounds = st.sidebar.slider("Number of Rounds", 1, 100, 18, step=1)
bet_fraction = st.sidebar.slider("Bet Fraction", 0.01, 1.0, 0.1, step=0.01)

# Run simulation when button is clicked
if st.sidebar.button("Run Simulation"):
    results, model_wealth, model_percentile, expected_values_over_time = simulate_betting_market(
        num_bettors=num_bettors,
        variance_ability=variance_ability,
        num_rounds=num_rounds,
        bet_fraction=bet_fraction
    )

    # Plot Weighted Average of Predictive Ability Over Time
    st.subheader("Weighted Average of Predictive Ability Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(expected_values_over_time)
    ax.set_ylim(0.5, 1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Weighted Average of Predictive Ability")
    ax.set_title("Weighted Average of Predictive Ability Over Time")
    st.pyplot(fig)

    # Plot Ability vs. End Wealth
    st.subheader("Ability vs. End Wealth (Log Scale)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results["Ability"], results["End Wealth"], alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel("Ability (Win Probability)")
    ax.set_ylabel("End Wealth (Log Scale)")
    ax.set_title("Ability vs. End Wealth After Betting Simulation (Log Scale)")
    st.pyplot(fig)

    # Plot Wealth Percentile vs. Ability Percentile
    st.subheader("Wealth Percentile vs. Ability Percentile")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results["Ability Percentile"], results["Wealth Percentile"], alpha=0.7, color="orange")
    ax.set_xlabel("Ability Percentile")
    ax.set_ylabel("Wealth Percentile")
    ax.set_title("Ability Percentile vs. Wealth Percentile After Betting Simulation")
    st.pyplot(fig)

    # Display Regression Summaries
    st.subheader("Regression Results")
    st.text("Regression Results for End Wealth:")
    st.text(model_wealth.summary())
    st.text("Regression Results for Wealth Percentile:")
    st.text(model_percentile.summary())
