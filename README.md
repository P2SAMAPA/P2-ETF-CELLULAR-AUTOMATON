# Cellular Automaton Market Engine

Models ETF interactions as a 1‑D cellular automaton (ring). Each ETF’s state is binary (return above/below median). The update rule (majority, Game of Life, threshold) simulates local influence. The per‑ETF score is the binary entropy of its state over many steps – a measure of predictability vs chaos.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Discretisation by median return
- Three rule types: majority, 1D Game of Life, threshold
- Score = binary entropy of state sequence (local entropy)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-cellular-automaton-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(steps × n))
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High entropy → ETF flips frequently – chaotic, may be a noise trader or regime‑sensitive.
- Low entropy → ETF remains stable – trending, predictable.
- This engine provides a unique perspective on market micro‑structure.

## Requirements

See `requirements.txt`.
