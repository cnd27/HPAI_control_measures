# A modelling assessment for the impact of control measures on highly pathogenic avian influenza transmission in poultry in Great Britain

This repository contains files for the analysis presented in the scientific paper "A modelling assessment for the impact of control measures on highly pathogenic avian influenza transmission in poultry in Great Britain" by Christopher N Davis, Edward M Hill, Chris P Jewell, Kristyna Rysava, Robin N Thompson and Michael J Tildesley 


Included python files are for MCMC fitting and model simulation for a between poultry premises transmission model for highly pathogenic avian influenza (HPAI).

---

## Python files

plot.py runs the model to generate the figures seen in the manuscript.

model.py is the model code used in plot.py.

matching_premises.py ensures consistency between data sets.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/cnd27/HPAI_control_measures.git
cd HPAI_control_measures
```

### 2. Set up dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python src/plot.py 
```

---

## Files in this repo

```
HPAI_control_measures/
├── data/                         # Input data
├── output/                       # Output files
├── src/                          # Python scripts
│   └── model.py                  # Main script
│   └── plot.py                   # Plotting script
│   └── matching_premises.py      # Data script
├── requirements.txt              # Dependencies
└── README.md                     # Instructions

```

---

## Requirements

See `requirements.txt` for full list and exact versions, but the key libraries used are:

- `geopandas`
- `matplotlib`
- `numpy`
- `pandas`
- `scipy` 

---

## Author

Christopher Davis
University of Warwick
GitHub: [@cnd27](https://github.com/cnd27)

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

