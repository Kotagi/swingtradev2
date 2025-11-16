# Installation Guide

## Quick Install

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Python Version Requirements

- **Python 3.11 or 3.12+** recommended
- Some packages (like `pandas-ta`) may require Python 3.12+

## Troubleshooting

### pandas-ta-classic (Python 3.11 Compatible)

The project uses `pandas-ta-classic` which is compatible with Python 3.11. The newer `pandas-ta` package requires Python 3.12+.

If you need to install it separately:
```bash
pip install pandas-ta-classic
```

### Alternative: Use Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Upgrade pip

If you encounter installation issues, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

## Individual Package Installation

If you prefer to install packages individually:

```bash
pip install pandas>=1.3
pip install numpy>=1.21
pip install yfinance>=0.2.0
pip install xgboost>=1.5.0
pip install scikit-learn>=1.0.0
pip install joblib>=1.0.0
pip install pyyaml>=5.4.0
pip install pyarrow>=20.0.0
pip install shap>=0.41.0
pip install matplotlib>=3.4
pip install pandas-ta-classic  # Compatible with Python 3.11+
```

## Verify Installation

After installation, verify key packages:

```bash
python -c "import pandas; import numpy; import yfinance; import xgboost; print('All core packages installed!')"
```

