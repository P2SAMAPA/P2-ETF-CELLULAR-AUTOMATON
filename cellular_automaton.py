import numpy as np

def discrete_state(returns_df):
    """
    Assign initial binary state to each ETF:
    1 if the ETF's average return over the window is above the median, else 0.
    """
    avg_returns = returns_df.mean(axis=0).values
    median = np.median(avg_returns)
    states = (avg_returns > median).astype(int)
    return states.tolist()

def elementary_ca_rule(rule_number):
    """Return a function that given (left, center, right) returns new state."""
    bits = [(rule_number >> i) & 1 for i in range(8)]
    def rule(l, c, r):
        idx = (l << 2) | (c << 1) | r
        return bits[7 - idx]
    return rule

def lempel_ziv_complexity(seq):
    """Lempel‑Ziv complexity (number of distinct substrings) normalized."""
    n = len(seq)
    if n == 0:
        return 0.0
    substrings = set()
    for i in range(n):
        for j in range(i+1, n+1):
            substrings.add(tuple(seq[i:j]))
    return len(substrings) / (n * (n+1) / 2)

def apply_cellular_automaton(states, rule_number=30, steps=50):
    n = len(states)
    rule = elementary_ca_rule(rule_number)
    history = [states.copy()]
    current = states.copy()
    for _ in range(steps):
        new = [0] * n
        for i in range(n):
            left = current[(i-1) % n]
            center = current[i]
            right = current[(i+1) % n]
            new[i] = rule(left, center, right)
        current = new
        history.append(current.copy())
    return current, history

def cellular_automaton_score(returns, rule_number=30, steps=50):
    """
    Return a stability score for each ETF: 1 - Lempel‑Ziv complexity.
    Higher score = more stable/predictable (trending).
    """
    states = discrete_state(returns)
    _, history = apply_cellular_automaton(states, rule_number, steps)
    history = np.array(history)  # shape (steps+1, n)
    scores = np.zeros(len(states))
    for i in range(len(states)):
        seq = history[:, i].tolist()
        complexity = lempel_ziv_complexity(seq)
        scores[i] = 1.0 - complexity   # stability = 1 - chaos
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}
