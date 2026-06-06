# Volatility Surface Causal Imputer

A causally clean implied volatility (IV) surface imputation engine designed for option backtesting. This tool reconstructs missing option IV surfaces without introducing any look-ahead bias (future information leakage) or relying on hardcoded dataset boundaries.

---

## Features

- **100% Causal Security**: Imputations are strictly chronological. There is zero future leakage from forward-fills, and the data-driven IV ceiling is computed as a running expanding maximum of historical rows up to time $T$.
- **Cross-Sectional Reconstruction**: Missing points within a smile are reconstructed using local polynomial (cubic) fits and linear interpolation across nearby liquid strikes.
- **Robust Wing Extrapolation**: Projecting out-of-the-money options beyond boundary strikes using linear trends coupled with exponential damping factors to prevent runaway volatility anomalies.
- **Auto-detected Parameters**: Auto-detects datetime formats, underlying spot/future index price columns, and option call/put metadata suffixes/strikes via generic regular expressions.
- **Built-in Validation Suite**: Includes automated checks to verify original data preservation, chronological sorting, completion, safety limits, and value plausibility (IQR outliers).

---

## File Structure

- `iv_imputer_v2.py`: The main imputation engine script.
- `dataset.csv`: The raw input dataset containing options IV columns (with gaps).
- `dataset_filled.csv`: The reconstructed dataset with all missing IVs filled.
- `submission_f_github.csv`: Format-compliant submission file containing only the reconstructed values for originally missing cells.

---

## Installation & Setup

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas scipy python-dateutil
```

---

## Usage

### Run Imputer on Default Dataset
If `dataset.csv` is in the current directory, simply run:

```bash
python iv_imputer_v2.py
```

### Run Imputer on a Custom Path
You can pass a custom dataset file as a command-line argument:

```bash
python iv_imputer_v2.py data/custom_dataset.csv
```

### Outputs Generated
- `dataset_filled.csv`: Complete, imputed dataset.
- `submission_f_github.csv` / `submission_f_github`: Extracted reconstructed entries in `id (datetime||column)` and `value` format.
- `submission.csv`: Standard target submission.
