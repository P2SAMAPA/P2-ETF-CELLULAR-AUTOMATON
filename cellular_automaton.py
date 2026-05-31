import numpy as np

def discrete_state(returns_df):
    """
    Initial binary state for each ETF based on individual rolling window:
    1 if the ETF's return on the last day > its average over the window, else 0.
    This gives more variation across ETFs.
    """
    last_return = returns_df.iloc[-1].values
    avg_return = returns_df.mean(axis=0).values
    states = (last_return > avg_return).astype(int)
    return states.tolist()

def elementary_ca_rule(rule_number):
    bits = [(rule_number >> i) & 1 for i in range(8)]
    def rule(l, c, r):
        idx = (l << 2) | (c << 1) | r
        return bits[7 - idx]
    return rule

def apply_cellular_automaton(states, rule_number=30, steps=100):
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

def lempel_ziv_complexity(seq):
    """Normalised Lempel‑Ziv complexity (number of distinct substrings)."""
    n = len(seq)
    if n == 0:
        return 0.0
    substrings = set()
    for i in range(n):
        for j in range(i+1, n+1):
            substrings.add(tuple(seq[i:j]))
    max_possible = n * (n+1) / 2
    return len(substrings) / max_possible

def cellular_automaton_score(returns, rule_number=30, steps=100):
    """
    For each ETF, compute stability = 1 - Lempel‑Ziv complexity of its state sequence.
    Lower complexity = more predictable.
    """
    states = discrete_state(returns)
    _, history = apply_cellular_automaton(states, rule_number, steps)
    history = np.array(history)
    scores = np.zeros(len(states))
    for i in range(len(states)):
        seq = history[:, i].tolist()
        complexity = lempel_ziv_complexity(seq)
        scores[i] = 1.0 - complexity   # stability
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}
