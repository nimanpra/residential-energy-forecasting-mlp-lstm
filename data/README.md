# Data

## Bureau of Meteorology (BOM)

Daily weather CSVs under `BOM/` are derived from **Australian Bureau of Meteorology** observations (Melbourne area), downloaded for the study period used in the paper. Public data; cite BOM when reusing.

## Residential smart meter data

| File | Description |
|------|-------------|
| `House 3_Melb East.csv` | 5-minute grid consumption (House 3, grid only) |
| `House 4_Melb West.csv` | 5-minute grid draw (House 4, net meter) |
| `House 4_Solar.csv` | 5-minute solar generation (House 4) |

These files were used in the Victoria University *Neural Networks and Deep Learning* course project and the subsequent paper. They represent real households in Melbourne; do not use them to identify individuals. For academic reuse, cite the paper repository and respect data ethics applicable in your jurisdiction.

## Layout expected by code

Scripts read `data/BOM/*.csv` and the three house CSVs from this directory (see `run_experiments.py`).
