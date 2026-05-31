import numpy as np

def discrete_state(returns_df):
    """
    Initial binary state for each ETF: 1 if last day's return > 0, else 0.
    This produces more varied initial patterns.
    """
    last_return = returns_df.iloc[-1].values
    states = (last_return > 0).astype(int)
    return states.tolist()

def elementary_ca_rule(rule_number):
    """Return a function that given (left, center, right) returns new state."""
    bits = [(rule_number >> i) & 1 for i in range(8)]
    def rule(l, c, r):
        idx = (l << 2) | (c << 1) | r
        return bits[7 - idx]
    return rule

def apply_cellular_automaton(states, rule_number=30, steps=100):
    """Evolve 1D elementary CA on a ring."""
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

def cellular_automaton_score(returns, rule_number=30, steps=100):
    """
    For each ETF, compute stability = 1 - flip_rate.
    Flip rate = (number of state changes) / (steps).
    Higher stability = more predictable / trending.
    """
    states = discrete_state(returns)
    _, history = apply_cellular_automaton(states, rule_number, steps)
    history = np.array(history)  # shape (steps+1, n)
    scores = np.zeros(len(states))
    for i in range(len(states)):
        seq = history[:, i]
        # count number of times the state changes between consecutive steps
        flips = np.sum(seq[:-1] != seq[1:])
        flip_rate = flips / steps
        scores[i] = 1.0 - flip_rate   # stability
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}
