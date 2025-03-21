import numpy as np
from multiprocessing import Pool


def single_simulation(S0, mu, sigma, T, dt):
    """Runs a single GBM simulation."""
    timesteps = int(T / dt)
    prices = np.zeros(timesteps)
    prices[0] = S0
    for t in range(1, timesteps):
        Z = np.random.standard_normal()
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return prices


def geometric_brownian_motion(S0, mu, sigma, T, dt, N):
    """Runs GBM simulations using multiprocessing for better performance."""
    with Pool() as pool:
        paths = pool.starmap(single_simulation, [(S0, mu, sigma, T, dt) for _ in range(N)])
    return np.array(paths)
