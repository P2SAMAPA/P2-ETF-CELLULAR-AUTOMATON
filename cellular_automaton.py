import numpy as np
from scipy.stats import entropy

def discrete_state(returns, n_states=2):
    """
    Convert returns to discrete states (0 or 1) based on median.
    Returns a numpy array.
    """
    median = np.median(returns)
    states = (returns > median).astype(int)
    # Convert to numpy array (values only) to avoid pandas index issues
    return states.values

def apply_cellular_automaton(states, rule_type='majority', threshold=0.5, steps=50):
    """
    Evolve the 1D cellular automaton for given steps.
    states: numpy array of ints (0/1)
    Returns the final state array and the full history (list of numpy arrays).
    """
    n = len(states)
    history = [states.copy()]
    current = states.copy()
    for _ in range(steps):
        new = np.zeros(n, dtype=int)
        for i in range(n):
            left = current[(i-1) % n]
            right = current[(i+1) % n]
            center = current[i]
            if rule_type == 'majority':
                # majority of three cells
                new[i] = 1 if (left + center + right) >= 2 else 0
            elif rule_type == 'game_of_life_1d':
                # 1D Game of Life: birth if exactly one neighbor is 1, survive if 1 and one neighbor, else die
                neighbors = left + right
                if center == 1:
                    new[i] = 1 if neighbors == 1 else 0
                else:
                    new[i] = 1 if neighbors == 1 else 0
            elif rule_type == 'threshold':
                # weighted average of neighbors (continuous), then threshold
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
    # For each cell, compute entropy of its state sequence over time
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
