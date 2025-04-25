import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import matplotlib.pyplot as plt

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price):
    intrinsic = max(S - K, 0)
    
    # Handle edge cases
    if market_price <= intrinsic:
        return 0.0  # IV can't be negative
    if T <= 0 or S <= 0:
        return np.nan
    sigma_min = 0.01
    sigma_max = 5.00
    
    try:
        # Use Brent's method which is more reliable
        return brentq(
            lambda sigma: black_scholes_call(S, K, T, r, sigma) - market_price,
            sigma_min, sigma_max,
            xtol=1e-6,
            maxiter=100
        )
    except:
        # Fallback to binary search if Brent fails
        for _ in range(50):
            sigma = (sigma_min + sigma_max) / 2
            price = black_scholes_call(S, K, T, r, sigma)
            if abs(price - market_price) < 0.01:
                return sigma
            if price > market_price:
                sigma_max = sigma
            else:
                sigma_min = sigma
        return (sigma_min + sigma_max) / 2 



T = 5/365  # 5 days to expiry
r = 0.00   # risk-free rate
iv_db = []
strikes = [9500, 9750, 10000, 10250, 10500]
option_prices = []



df = pd.read_csv('round-3-island-data-bottle/prices_round_3_day_0.csv', delimiter=';')
df.columns = df.columns.str.strip() 
rock = df[df['product'] == 'VOLCANIC_ROCK']
stock_prices = rock['mid_price'].values
for strike in strikes:
    coupon = df[df['product'] == f'VOLCANIC_ROCK_VOUCHER_{strike}']
    option_prices_single_voucher = coupon['mid_price'].values
    option_prices.append(option_prices_single_voucher)

for i, strike in enumerate(strikes):
    strike_vol = []
    for j, stock_price in enumerate(stock_prices):
        iv = implied_volatility(stock_price, strike, T, r, option_prices[i][j])
        strike_vol.append(iv)
    iv_db.append(strike_vol)

for i, strike in enumerate(strikes):
    print(f'Average volatility of {strike} strike option: ', np.mean(iv_db[i]))
    plt.figure
    plt.plot(iv_db[i], 'b-', linewidth=2)
    plt.title(f'volatility of {strike} call option')
    plt.savefig(f'volatility_curves/vol_{strike}.png', dpi=300)
