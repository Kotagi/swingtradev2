import yaml
import pytest
import pandas as pd
import yfinance as yf

from features.registry import load_enabled_features

# Load only the tickers we care about
with open("config/test_tickers.yaml") as f:
    TEST_TICKERS = yaml.safe_load(f)["test_tickers"]

@pytest.mark.parametrize("ticker", TEST_TICKERS)
def test_pipeline_features_on_real_data(ticker):
    df = yf.download(ticker, period="30d", progress=False)
    assert not df.empty, f"No data for {ticker}"

    features = load_enabled_features("config/features.yaml")
    assert features, "No features enabled!"

    for name, func in features.items():
        out = func(df)
        assert isinstance(out, pd.Series), f"{name} did not return a Series"
        assert out.dtype == float, f"{name} dtype is {out.dtype}"
        if len(df) >= 10:
            assert out.notna().any(), f"{name} returned all NaN for {ticker}"
