import numpy as np
from scipy.stats import entropy

def discrete_state(returns_df):
    """
    Convert each ETF's average return over the window to binary state (0 or 1)
    based on whether it is above the median of all ETFs' average returns.
    Returns a list of ints.
    """
    avg_returns = returns_df.mean(axis=0).values
    median = np.median(avg_returns)
    states = (avg_returns > median).astype(int)
    return states.tolist()

def apply_cellular_automaton(states, rule_type='majority', threshold=0.5, steps=50):
    """
    Evolve the 1D cellular automaton for given steps.
    states: list of ints (0/1)
    Returns final state list and full history (list of lists).
    """
    n = len(states)
    history = [states.copy()]
    current = states.copy()
    for _ in range(steps):
        new = [0] * n
        for i in range(n):
            left = current[(i-1) % n]
            right = current[(i+1) % n]
            center = current[i]
            if rule_type == 'majority':
                new[i] = 1 if (left + center + right) >= 2 else 0
            elif rule_type == 'game_of_life_1d':
                neighbors = left + right
                if center == 1:
                    new[i] = 1 if neighbors == 1 else 0
                else:
                    new[i] = 1 if neighbors == 1 else 0
            elif rule_type == 'threshold':
                avg = (left + center + right) / 3.0
                new[i] = 1 if avg > threshold else 0
            else:
                raise ValueError(f"Unknown rule: {rule_type}")
        current = new
        history.append(current.copy())
    return current, history

def cellular_automaton_score(returns, rule_type='majority', threshold=0.5, steps=50):
    """
    For each ETF, compute the entropy of its state over time (local entropy).
    High entropy = chaotic influence, low entropy = stable.
    """
    states = discrete_state(returns)
    _, history = apply_cellular_automaton(states, rule_type, threshold, steps)
    history = np.array(history)  # shape (steps+1, n)
    scores = np.zeros(len(states))
    for i in range(len(states)):
        seq = history[:, i]
        p0 = np.mean(seq == 0)
        p1 = 1 - p0
        if p0 == 0 or p1 == 0:
            scores[i] = 0.0
        else:
            scores[i] = entropy([p0, p1], base=2)
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}
