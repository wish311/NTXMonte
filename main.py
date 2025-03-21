#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_fetch import fetch_stock_data, calculate_mu_sigma
from gbm import geometric_brownian_motion

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"[+] Saved simulation results to {filename}")

def main():
    parser = argparse.ArgumentParser(description="GBM Stock Price Simulator")
    parser.add_argument("symbol", nargs="?", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("symbol", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--simulations", type=int, default=250000, help="Number of simulations")
    parser.add_argument("--days", type=int, default=252, help="Trading days to simulate")
    parser.add_argument("--plot", action="store_true", help="Show simulation plot")
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV")

    args = parser.parse_args()

    if not args.symbol:
        print("[-] Error: You must provide a stock symbol.\n")
        parser.print_help()
        exit(1)




    print(f"[+] Fetching data for {args.symbol}...")
    prices = fetch_stock_data(args.symbol, args.start, args.end)
    if prices is None or prices.empty:
        print("[-] No data found. Check the ticker symbol and date range.")
        return

    mu, sigma = calculate_mu_sigma(prices)
    S0 = float(prices.iloc[-1])  # Convert to a scalar float


    T = args.days / 252
    dt = 1 / 252

    print(f"[+] Running {args.simulations} simulations...")
    paths = geometric_brownian_motion(S0, mu, sigma, T, dt, args.simulations)

    terminal_prices = paths[:, -1]
    print(f"\nSimulation Results for {args.symbol}:")
    print(f"  Start Price: {S0:.2f}")
    print(f"  Mean Final Price: {np.mean(terminal_prices):.2f}")
    print(f"  5th Percentile: {np.percentile(terminal_prices, 5):.2f}")
    print(f"  95th Percentile: {np.percentile(terminal_prices, 95):.2f}")

    if args.save_csv:
        save_to_csv(terminal_prices, f"{args.symbol}_simulation.csv")

    if args.plot:
        for i in range(10):
            plt.plot(paths[i])
        plt.title(f"{args.symbol} Sample GBM Paths")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
