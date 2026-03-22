# Research notes (supplementary)

## Data inventory

| Asset | Role |
|-------|------|
| `House 3_Melb East.csv` | 5-minute grid consumption (grid-only household) |
| `House 4_Melb West.csv` | 5-minute grid draw |
| `House 4_Solar.csv` | 5-minute solar generation |
| `BOM/*.csv` | Daily BOM weather for overlap with smart-meter period |

## Experimental outcomes (summary)

Reported metrics are in `results.json` and the paper. MLP vs LSTM behaviour differs strongly between households; persistence baselines are included for 5-minute resolution.

## Reproducibility

- Random seed: `42` in `run_experiments.py`
- Train/test split and sequence length are defined in the script and stored in `results.json`

## Future experiments (not required for current paper)

- Hourly or sub-daily weather aligned to 5-minute load
- Additional households or public datasets (e.g. Ausgrid) for generalisation
- Optional: multiple random seeds for variance bands
