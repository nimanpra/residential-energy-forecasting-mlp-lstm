# Residential energy forecasting: MLP vs LSTM (Melbourne)

Code and data for the study **Beyond Weather Correlation: A Comparative Study of Static and Temporal Neural Architectures for Fine-Grained Residential Energy Consumption Forecasting in Melbourne, Australia**.

**Authors:** Prasad Nimantha Madusanka Ukwatta Hewage, Hao Wu — School of Computing and Information Technology, Victoria University, Sydney, Australia.

## Contents

| Path | Description |
|------|-------------|
| `paper.tex` / `paper.pdf` | Manuscript (LaTeX source + compiled PDF) |
| `arxiv_submission.tar.gz` | LaTeX bundle for arXiv upload |
| `data/` | BOM weather CSVs + household smart meter CSVs (see `data/README.md`) |
| `run_experiments.py` | Full pipeline: loads data, trains MLP (sklearn) and LSTM (TensorFlow), baselines, writes `results.json` and figures |
| `generate_figures.py` | Regenerates figures from saved metrics (lighter than full retrain) |
| `figures/` | Publication figures (PDF + PNG) |
| `results.json` | Metrics from the last full run |
| `docs/` | Submission notes and checklist |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce experiments

From the repository root:

```bash
python run_experiments.py
```

Training uses a fixed random seed (`42`) for reproducibility. Runtime depends on CPU/GPU; TensorFlow is required for LSTM.

To refresh figures only (uses `results.json` / embedded metrics in script):

```bash
python generate_figures.py
```

## Citation

If you use this repository or the associated paper, cite the PDF author list and year when published, or this repository URL until a DOI is assigned.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The manuscript remains under the license you choose for the PDF/preprint. Third-party data (BOM) remains subject to BOM terms; household CSVs are shared for research reproducibility as described in `data/README.md`.
