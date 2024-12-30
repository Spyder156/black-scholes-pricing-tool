# Step 1: Python REPL Application
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import mysql.connector

def black_scholes(S, K, T, r, sigma, option_type):
    """Calculate Black-Scholes option price."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return price

# Step 2: GUI Layer
def streamlit_app():
    st.title("Black-Scholes Option Pricing Tool")

    # User inputs
    S = st.number_input("Asset price (S)", value=100.0, step=0.1)
    K = st.number_input("Strike price (K)", value=100.0, step=0.1)
    T = st.number_input("Time to expiry (T in years)", value=1.0, step=0.1)
    r = st.number_input("Risk-free interest rate (r)", value=0.05, step=0.01)
    sigma = st.number_input("Volatility (sigma)", value=0.2, step=0.01)
    option_type = st.selectbox("Option type", ["call", "put"])
    purchase_price = st.number_input("Purchase price of the option", value=0.0, step=0.1)

    # Calculate option price
    price = black_scholes(S, K, T, r, sigma, option_type)
    st.write(f"The {option_type} price is: {price:.2f}")

    # Step 3: Heatmap and PnL
    if st.button("Generate Heatmap"):
        vol_range = np.linspace(0.1, 0.5, 10)
        price_range = np.linspace(50, 150, 10)

        pnl_values = []
        for vol in vol_range:
            row = []
            for sp in price_range:
                option_value = black_scholes(sp, K, T, r, vol, option_type)
                pnl = option_value - purchase_price
                row.append(pnl)
            pnl_values.append(row)

        df = pd.DataFrame(pnl_values, index=vol_range, columns=price_range)
        st.write("PnL Heatmap:")
        fig, ax = plt.subplots()
        sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
        plt.xlabel("Stock Price")
        plt.ylabel("Volatility")
        st.pyplot(fig)

    # Step 4: Save Inputs and Outputs to MySQL
    if st.button("Save Calculation"):
        save_to_database(S, K, T, r, sigma, option_type, purchase_price, df)
        st.success("Calculation saved to database.")

# Step 5: Database Integration
def save_to_database(S, K, T, r, sigma, option_type, purchase_price, df):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="your_username",
            password="your_password",
            database="black_scholes_db"
        )
        cursor = conn.cursor()

        # Save inputs
        input_query = """
            INSERT INTO inputs_table (asset_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        input_values = (S, K, T, r, sigma, option_type)
        cursor.execute(input_query, input_values)
        calculation_id = cursor.lastrowid

        # Save outputs
        for vol, row in zip(df.index, df.values):
            for sp, pnl in zip(df.columns, row):
                output_query = """
                    INSERT INTO outputs_table (volatility, stock_price, pnl, calculation_id)
                    VALUES (%s, %s, %s, %s)
                """
                output_values = (vol, sp, pnl, calculation_id)
                cursor.execute(output_query, output_values)

        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")

# Uncomment the line below to run the Streamlit app
# streamlit_app()
