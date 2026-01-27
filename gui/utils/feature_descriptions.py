"""
Feature Descriptions Module

Provides descriptions and interpretation guidance for technical indicators
used in the swing trading ML pipeline.
"""

FEATURE_DESCRIPTIONS = {
    # Price Features
    "price_vs_ma200": {
        "name": "Price vs MA200",
        "description": (
            "Price normalized relative to 200-day moving average. "
            "This feature provides long-term context for current price level. "
            "Values > 1.0 indicate price above long-term average (bullish), "
            "while values < 1.0 indicate price below long-term average (bearish)."
        ),
        "interpretation": (
            "Normalized values are ratios. Values > 1.0 indicate price is above the 200-day moving average (bullish trend), "
            "while values < 1.0 indicate price is below the 200-day moving average (bearish trend). "
            "Typical filter thresholds: > 1.05 for strong bullish conditions, < 0.95 for bearish conditions."
        )
    },
    
    # Momentum Features
    "rsi14": {
        "name": "RSI (14)",
        "description": (
            "Relative Strength Index (14-period) with centered normalization. "
            "Classic momentum oscillator for identifying overbought/oversold conditions. "
            "RSI measures the speed and magnitude of price changes."
        ),
        "interpretation": (
            "Normalized values are centered to [-1, +1] range. "
            "Values > 0 indicate RSI above 50 (potentially overbought), "
            "values < 0 indicate RSI below 50 (potentially oversold). "
            "Typical filter thresholds: > 0.5 (RSI > 75) for overbought conditions, "
            "< -0.5 (RSI < 25) for oversold conditions."
        )
    },
    
    # Return Features
    "daily_return": {
        "name": "Daily Return",
        "description": (
            "Daily percentage return calculated as the percentage change from previous day's close. "
            "Captures daily momentum and price changes. Values are clipped to ±20% to prevent outliers."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.2, 0.2] range (±20%). "
            "Positive values indicate price increase from previous day (bullish), "
            "negative values indicate price decrease (bearish). "
            "Typical filter thresholds: > 0.05 (5% gain) for strong bullish momentum, "
            "< -0.05 (-5% loss) for bearish momentum."
        )
    },
    
    # Volatility Features
    "atr14_normalized": {
        "name": "ATR (14)",
        "description": (
            "Average True Range (14-day) normalized by price. "
            "Measures volatility based on true range, accounting for gaps and overnight moves. "
            "More accurate volatility measure than simple price ranges."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.0, 0.2] range (0% to 20% of price). "
            "Higher values indicate higher volatility (wider price swings), "
            "lower values indicate lower volatility (tighter price ranges). "
            "Typical filter thresholds: > 0.05 (5% volatility) for high volatility stocks, "
            "< 0.02 (2% volatility) for low volatility stocks."
        )
    },
    
    # Volume Features
    "relative_volume": {
        "name": "Relative Volume",
        "description": (
            "Current volume relative to 20-day average volume. "
            "Identifies unusual volume activity, which is important for breakout detection. "
            "Higher values indicate above-average trading activity."
        ),
        "interpretation": (
            "Normalized values use log1p transformation after clipping to [0, 10] range. "
            "Values > 1.0 indicate above-average volume (increased interest), "
            "values < 1.0 indicate below-average volume (decreased interest). "
            "Typical filter thresholds: > 1.5 for high volume breakouts, "
            "< 0.5 for low volume consolidation."
        )
    },
    
    # Additional Price Features
    "price": {
        "name": "Price",
        "description": (
            "Raw closing price value. Useful for filtering by price ranges. "
            "Not normalized - use for filtering, not ML input."
        ),
        "interpretation": (
            "Raw price values in dollars. Use for price-based filtering (e.g., $1-$5, >$5, >$10). "
            "Not normalized, so values vary widely by stock price."
        )
    },
    "price_log": {
        "name": "Log Price",
        "description": (
            "Log-transformed closing price. Compresses large differences between high and low priced stocks, "
            "making price comparable across different price ranges."
        ),
        "interpretation": (
            "Log-transformed values. Natural log compression makes values more comparable across different price ranges. "
            "Useful for ML models that need normalized inputs."
        )
    },
    
    # Additional Return Features
    "gap_pct": {
        "name": "Gap %",
        "description": (
            "Gap percentage between today's open and yesterday's close. "
            "Measures overnight momentum and gap trading opportunities. "
            "Positive values indicate gap-up (bullish), negative values indicate gap-down (bearish)."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.2, 0.2] range (±20%). "
            "Positive values indicate gap-up (bullish), negative values indicate gap-down (bearish). "
            "Typical filter thresholds: > 0.02 (2% gap-up) for bullish momentum, "
            "< -0.02 (-2% gap-down) for bearish momentum."
        )
    },
    "weekly_return_5d": {
        "name": "Weekly Return (5d)",
        "description": (
            "5-day (weekly) percentage return. Captures short-term momentum over approximately one week. "
            "Values are clipped to ±30% to prevent outliers."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.3, 0.3] range (±30%). "
            "Positive values indicate price increase over 5 days (bullish), "
            "negative values indicate price decrease (bearish). "
            "Typical filter thresholds: > 0.10 (10% weekly gain) for strong momentum."
        )
    },
    "monthly_return_21d": {
        "name": "Monthly Return (21d)",
        "description": (
            "21-day (monthly) percentage return. Captures medium-term momentum over approximately one month. "
            "Values are clipped to ±50% to prevent outliers."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.5, 0.5] range (±50%). "
            "Positive values indicate price increase over 21 days (bullish), "
            "negative values indicate price decrease (bearish). "
            "Typical filter thresholds: > 0.15 (15% monthly gain) for strong momentum."
        )
    },
    "quarterly_return_63d": {
        "name": "Quarterly Return (63d)",
        "description": (
            "63-day (quarterly) percentage return. Captures longer-term momentum over approximately one quarter. "
            "Values are clipped to ±100% to prevent outliers."
        ),
        "interpretation": (
            "Normalized values are clipped to [-1.0, 1.0] range (±100%). "
            "Positive values indicate price increase over 63 days (bullish), "
            "negative values indicate price decrease (bearish). "
            "Typical filter thresholds: > 0.20 (20% quarterly gain) for strong momentum."
        )
    },
    "ytd_return": {
        "name": "Year-to-Date Return",
        "description": (
            "Year-to-Date percentage return. Measures performance from start of year. "
            "Values are clipped to [-100%, +200%] to prevent extreme outliers."
        ),
        "interpretation": (
            "Normalized values are clipped to [-1.0, 2.0] range (-100% to +200%). "
            "Positive values indicate positive YTD performance, "
            "negative values indicate negative YTD performance. "
            "Typical filter thresholds: > 0.10 (10% YTD gain) for strong performers."
        )
    },
    
    # 52-Week Features
    "dist_52w_high": {
        "name": "Distance from 52W High",
        "description": (
            "Distance from 52-week high. Measures how far price is from recent highs, "
            "useful for breakout detection. 0.0 means at 52-week high, negative values mean below high."
        ),
        "interpretation": (
            "Normalized values are clipped to [-1.0, 0.5] range. "
            "0.0 indicates at 52-week high, negative values indicate below high (more negative = further below). "
            "Typical filter thresholds: > -0.05 (within 5% of 52W high) for breakout candidates."
        )
    },
    "dist_52w_low": {
        "name": "Distance from 52W Low",
        "description": (
            "Distance from 52-week low. Measures how far price is from recent lows, "
            "useful for support detection. 0.0 means at 52-week low, positive values mean above low."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.5, 2.0] range. "
            "0.0 indicates at 52-week low, positive values indicate above low (more positive = further above). "
            "Typical filter thresholds: < 0.20 (within 20% of 52W low) for support bounces."
        )
    },
    "pos_52w": {
        "name": "Position in 52W Range",
        "description": (
            "Position within 52-week range (0=low, 1=high). Normalized position within yearly range, "
            "useful for relative strength analysis."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates at 52-week low, 1.0 indicates at 52-week high, 0.5 indicates midpoint. "
            "Typical filter thresholds: > 0.80 (top 20% of range) for strong relative strength."
        )
    },
    
    # Moving Average Features
    "sma20_ratio": {
        "name": "SMA20 Ratio",
        "description": (
            "Price relative to 20-day Simple Moving Average. Short-term trend indicator. "
            "Values > 1.0 indicate price above SMA20 (bullish), values < 1.0 indicate below (bearish)."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.5, 1.5] range. "
            "1.0 indicates price equals SMA20, > 1.0 indicates above (bullish), < 1.0 indicates below (bearish). "
            "Typical filter thresholds: > 1.02 for bullish short-term trend."
        )
    },
    "sma50_ratio": {
        "name": "SMA50 Ratio",
        "description": (
            "Price relative to 50-day Simple Moving Average. Medium-term trend indicator. "
            "Values > 1.0 indicate price above SMA50 (bullish), values < 1.0 indicate below (bearish)."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.5, 1.5] range. "
            "1.0 indicates price equals SMA50, > 1.0 indicates above (bullish), < 1.0 indicates below (bearish). "
            "Typical filter thresholds: > 1.02 for bullish medium-term trend."
        )
    },
    "sma200_ratio": {
        "name": "SMA200 Ratio",
        "description": (
            "Price relative to 200-day Simple Moving Average. Long-term trend indicator, "
            "classic bull/bear market signal. Values > 1.0 indicate long-term uptrend."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.5, 2.0] range. "
            "1.0 indicates price equals SMA200, > 1.0 indicates above (bullish long-term), < 1.0 indicates below (bearish). "
            "Typical filter thresholds: > 1.05 for strong long-term uptrend."
        )
    },
    "sma20_sma50_ratio": {
        "name": "SMA20/SMA50 Ratio",
        "description": (
            "Ratio of SMA20 to SMA50 (moving average crossover). "
            "Values > 1.0 indicate SMA20 above SMA50 (bullish crossover, uptrend), "
            "values < 1.0 indicate bearish crossover (downtrend)."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.8, 1.2] range. "
            "1.0 indicates SMA20 equals SMA50, > 1.0 indicates bullish crossover, < 1.0 indicates bearish crossover. "
            "Typical filter thresholds: > 1.01 for bullish short-to-medium term trend."
        )
    },
    "sma50_sma200_ratio": {
        "name": "SMA50/SMA200 Ratio",
        "description": (
            "Ratio of SMA50 to SMA200 (Golden Cross/Death Cross). Classic moving average crossover indicator. "
            "Values > 1.0 indicate Golden Cross (bullish long-term trend), < 1.0 indicate Death Cross (bearish)."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.6, 1.4] range. "
            "1.0 indicates SMA50 equals SMA200, > 1.0 indicates Golden Cross (bullish), < 1.0 indicates Death Cross (bearish). "
            "Typical filter thresholds: > 1.02 for strong long-term bullish trend."
        )
    },
    "sma50_slope": {
        "name": "SMA50 Slope",
        "description": (
            "5-day change in SMA50, normalized by price. Measures rate of change (slope) of medium-term trend. "
            "Positive values indicate SMA50 rising (bullish momentum), negative values indicate falling (bearish)."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.1, 0.1] range. "
            "0.0 indicates SMA50 is flat, > 0.0 indicates rising (bullish), < 0.0 indicates falling (bearish). "
            "Typical filter thresholds: > 0.01 for bullish medium-term momentum."
        )
    },
    "sma200_slope": {
        "name": "SMA200 Slope",
        "description": (
            "10-day change in SMA200, normalized by price. Measures rate of change (slope) of long-term trend. "
            "Positive values indicate SMA200 rising (bullish long-term momentum), negative values indicate falling (bearish)."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.1, 0.1] range. "
            "0.0 indicates SMA200 is flat, > 0.0 indicates rising (bullish), < 0.0 indicates falling (bearish). "
            "Typical filter thresholds: > 0.005 for bullish long-term momentum."
        )
    },
    "kama_slope": {
        "name": "KAMA Slope",
        "description": (
            "Kaufman Adaptive Moving Average Slope - adaptive trend strength indicator. "
            "Adapts to market efficiency: fast in trends, slow in noise. More responsive than SMA in efficient markets."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.1, 0.1] range. "
            "0.0 indicates KAMA is flat, > 0.0 indicates rising (adaptive bullish momentum), < 0.0 indicates falling (bearish). "
            "Typical filter thresholds: > 0.01 for adaptive bullish momentum."
        )
    },
    
    # Additional Volatility Features
    "volatility_5d": {
        "name": "Volatility (5d)",
        "description": (
            "5-day rolling standard deviation of daily returns. Captures short-term price volatility and risk. "
            "Higher values indicate more volatile price movements."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.0, 0.15] range (0% to 15% daily volatility). "
            "Higher values indicate higher volatility (wider price swings). "
            "Typical filter thresholds: > 0.03 (3% volatility) for high volatility stocks."
        )
    },
    "volatility_21d": {
        "name": "Volatility (21d)",
        "description": (
            "21-day rolling standard deviation of daily returns. Captures medium-term price volatility and risk. "
            "Higher values indicate more volatile price movements."
        ),
        "interpretation": (
            "Normalized values are clipped to [0.0, 0.15] range (0% to 15% daily volatility). "
            "Higher values indicate higher volatility (wider price swings). "
            "Typical filter thresholds: > 0.03 (3% volatility) for high volatility stocks."
        )
    },
    "volatility_ratio": {
        "name": "Volatility Ratio",
        "description": (
            "Short-term volatility (5-day) vs long-term volatility (21-day). "
            "Identifies volatility expansion/compression regimes. "
            "Values > 1 indicate volatility expanding, < 1 indicate contracting."
        ),
        "interpretation": (
            "Normalized values are clipped to [0, 2] range. "
            "> 1.0 indicates volatility expanding (short-term vol > long-term vol), "
            "< 1.0 indicates volatility contracting. "
            "Typical filter thresholds: > 1.2 for volatility expansion (breakout conditions)."
        )
    },
    "bollinger_band_width": {
        "name": "Bollinger Band Width",
        "description": (
            "Bollinger Band Width (log normalized) - measures volatility compression/expansion. "
            "One of the best predictors of breakouts. BB squeezes often precede strong 10-30 day swings."
        ),
        "interpretation": (
            "Normalized values use log1p transformation. "
            "Lower values indicate volatility compression (squeeze), higher values indicate expansion. "
            "Typical filter thresholds: < 0.1 for squeeze conditions (breakout potential)."
        )
    },
    "ttm_squeeze_on": {
        "name": "TTM Squeeze",
        "description": (
            "TTM Squeeze condition - binary flag indicating volatility contraction (squeeze). "
            "This is the #1 breakout indicator used by quant retail traders. "
            "Identifies tight trading ranges that often precede breakouts."
        ),
        "interpretation": (
            "Binary values: 0 = no squeeze, 1 = squeeze active (Bollinger Bands inside Keltner Channels). "
            "Filter for value = 1 to identify high-probability breakout setups."
        )
    },
    "ttm_squeeze_momentum": {
        "name": "TTM Squeeze Momentum",
        "description": (
            "TTM Squeeze momentum (normalized) - momentum direction during squeeze. "
            "Helps identify which direction the breakout is likely to occur. "
            "Positive values indicate price above SMA20 (bullish momentum)."
        ),
        "interpretation": (
            "Normalized values are normalized by price. "
            "Positive values indicate bullish momentum (price above SMA20), "
            "negative values indicate bearish momentum (price below SMA20). "
            "Typical filter thresholds: > 0.01 for bullish breakout direction."
        )
    },
    "volatility_of_volatility": {
        "name": "Volatility of Volatility",
        "description": (
            "Measures how unstable volatility itself is (meta volatility). "
            "Tells model whether volatility indicators are reliable or chaotic. "
            "Low values indicate calm, stable regime; high values indicate chaotic regime."
        ),
        "interpretation": (
            "Normalized values are clipped to [0, 3] range (relative to long-term average volatility). "
            "Low values (0-1) indicate stable regime (signals behave cleanly), "
            "high values (>2) indicate chaotic regime (risk of whipsaws). "
            "Typical filter thresholds: < 1.5 for stable, tradable conditions."
        )
    },
    
    # Additional Volume Features
    "log_volume": {
        "name": "Log Volume",
        "description": (
            "Log-transformed volume. Handles zero volumes gracefully and compresses wide range of volume values, "
            "making volume comparable across stocks."
        ),
        "interpretation": (
            "Normalized values use log1p transformation. Compresses wide range of volume values. "
            "Useful for ML models but less intuitive for filtering. "
            "Consider using relative_volume instead for filtering."
        )
    },
    "log_avg_volume_20d": {
        "name": "Log Avg Volume (20d)",
        "description": (
            "Log-transformed 20-day average volume. Provides smoothed, normalized view of volume trends. "
            "Medium-term volume baseline."
        ),
        "interpretation": (
            "Normalized values use log1p transformation. Smoothed view of volume trends. "
            "Useful for ML models but less intuitive for filtering. "
            "Consider using relative_volume instead for filtering."
        )
    },
    "chaikin_money_flow": {
        "name": "Chaikin Money Flow",
        "description": (
            "Chaikin Money Flow (20-period) - accumulation vs distribution. "
            "Combines price action with volume. One of the strongest predictors of future swings. "
            "Exposes quiet accumulation before breakouts."
        ),
        "interpretation": (
            "Normalized values are in [-1.0, 1.0] range. "
            "-1.0 indicates strong selling pressure/distribution, "
            "+1.0 indicates strong buying pressure/accumulation, "
            "0.0 indicates neutral. "
            "Typical filter thresholds: > 0.2 for accumulation (bullish), < -0.2 for distribution (bearish)."
        )
    },
    "obv_momentum": {
        "name": "OBV Momentum",
        "description": (
            "OBV Momentum (OBV Rate of Change) - 10-day percentage change of On-Balance Volume. "
            "Gives volume acceleration, not just volume level. Catches institutional volume movements before price shifts."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.5, 0.5] range (±50% change). "
            "Positive values indicate volume accelerating upward (bullish), "
            "negative values indicate volume accelerating downward (bearish). "
            "Typical filter thresholds: > 0.1 (10% OBV increase) for bullish volume momentum."
        )
    },
    
    # Additional Momentum Features
    "macd_histogram_normalized": {
        "name": "MACD Histogram",
        "description": (
            "MACD Histogram normalized by price - measures momentum acceleration/deceleration. "
            "The most predictive part of MACD. Detects early momentum shifts and trend strength."
        ),
        "interpretation": (
            "Normalized values are normalized by price (histogram / close). "
            "Positive values indicate bullish momentum acceleration, "
            "negative values indicate bearish momentum deceleration. "
            "Typical filter thresholds: > 0.001 for bullish momentum acceleration."
        )
    },
    "ppo_histogram": {
        "name": "PPO Histogram",
        "description": (
            "PPO (Percentage Price Oscillator) Histogram - percentage-based momentum acceleration/deceleration. "
            "Scale-invariant and cross-ticker comparable. Measures momentum in percent (unlike MACD which is in absolute price units)."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.2, 0.2] range. "
            "Positive values indicate PPO accelerating above signal (bullish momentum acceleration), "
            "negative values indicate PPO decelerating below signal (bearish). "
            "Typical filter thresholds: > 0.01 for bullish momentum acceleration."
        )
    },
    "dpo": {
        "name": "DPO",
        "description": (
            "DPO (Detrended Price Oscillator, 20-period) - cyclical indicator that removes long-term trend. "
            "Highlights short-term price cycles. Perfect for identifying mean-reversion zones."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.2, 0.2] range. "
            "Positive values indicate price above detrended average (cycle peak, overextended), "
            "negative values indicate price below detrended average (cycle trough, compressed). "
            "Typical filter thresholds: < -0.05 for oversold (compressed) conditions."
        )
    },
    "roc10": {
        "name": "ROC (10)",
        "description": (
            "ROC (Rate of Change) 10-period - short-term momentum velocity indicator. "
            "Highly predictive in breakouts and pullbacks. Captures velocity, not just simple percent change."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.5, 0.5] range (±50% change over 10 periods). "
            "Positive values indicate price rising over 10 periods (bullish momentum), "
            "negative values indicate price falling (bearish). "
            "Typical filter thresholds: > 0.10 (10% ROC) for strong bullish momentum."
        )
    },
    "roc20": {
        "name": "ROC (20)",
        "description": (
            "ROC (Rate of Change) 20-period - medium-term momentum velocity indicator. "
            "Highly predictive in breakouts and pullbacks. Captures velocity, not just simple percent change."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.7, 0.7] range (±70% change over 20 periods). "
            "Positive values indicate price rising over 20 periods (bullish momentum), "
            "negative values indicate price falling (bearish). "
            "Typical filter thresholds: > 0.15 (15% ROC) for strong bullish momentum."
        )
    },
    "stochastic_k14": {
        "name": "Stochastic %K (14)",
        "description": (
            "Stochastic Oscillator %K (14-period) - position within trading range. "
            "Directly captures overbought/oversold relative to range. Better than RSI in many trend scenarios."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates close at lowest low (extremely oversold), "
            "1.0 indicates close at highest high (extremely overbought), "
            "0.5 indicates close in middle of range. "
            "Typical filter thresholds: < 0.2 for oversold, > 0.8 for overbought."
        )
    },
    "cci20": {
        "name": "CCI (20)",
        "description": (
            "CCI (Commodity Channel Index, 20-period) - standardized distance from trend oscillator. "
            "Hybrid indicator combining momentum, volatility, and mean reversion. "
            "Captures trend exhaustion & reversion points."
        ),
        "interpretation": (
            "Normalized values use tanh compression, typically in [-0.76, 0.76] range. "
            "High values (>0.5) indicate momentum burst, overbought, "
            "low values (<-0.5) indicate selling pressure, oversold, "
            "near 0 indicates price near typical mean. "
            "Typical filter thresholds: < -0.5 for oversold, > 0.5 for overbought."
        )
    },
    "williams_r14": {
        "name": "Williams %R (14)",
        "description": (
            "Williams %R (14-period) - range momentum/reversion oscillator. "
            "Very sensitive to reversal points. Strong complement to RSI and Stochastic. "
            "Helps catch swing entries inside trends."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates close at highest high (extremely overbought), "
            "1.0 indicates close at lowest low (extremely oversold), "
            "0.5 indicates close in middle of range. "
            "Typical filter thresholds: > 0.8 for oversold (reversal potential), < 0.2 for overbought."
        )
    },
    
    # Market Context
    "beta_spy_252d": {
        "name": "Beta vs SPY (252d)",
        "description": (
            "Rolling beta vs SPY over 252 trading days. Measures stock's sensitivity to market movements. "
            "Provides market context - how stock moves relative to overall market."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates Beta = -1 (inverse correlation), "
            "0.5 indicates Beta = 1 (moves with market), "
            "1.0 indicates Beta = 3 (highly sensitive to market). "
            "Typical filter thresholds: > 0.4 for market-sensitive stocks."
        )
    },
    "mkt_spy_dist_sma200": {
        "name": "SPY Distance from SMA200",
        "description": (
            "SPY distance from SMA200 (market extension vs long-term trend). "
            "Measures how extended the market is vs its long-term trend baseline. "
            "Provides market regime context: risk-on (bullish) vs risk-off (bearish) environment. "
            "Z-score normalized over 1260 days (~5 years) to make it comparable across different market regimes."
        ),
        "interpretation": (
            "Normalized values are z-scores clipped to [-3, 3] range. "
            "Higher (positive) values indicate risk-on/bullish environment (market extended above trend), "
            "near 0 indicates neutral (market at trend baseline), "
            "lower (negative) values indicate risk-off/bearish regime (market below trend). "
            "Typical filter thresholds: > 1.0 for strong bullish market conditions, "
            "< -1.0 for bearish market conditions."
        )
    },
    "relative_strength_vs_sector": {
        "name": "Relative Strength vs. Sector",
        "description": (
            "Stock's 20-day return compared to its sector ETF's 20-day return. "
            "Measures whether a stock is outperforming or underperforming its sector. "
            "Positive values indicate the stock is leading its sector (bullish), "
            "negative values indicate lagging (bearish). "
            "Uses actual sector ETFs (XLK for Technology, XLF for Financials, etc.) "
            "with historical fallbacks for newer sectors."
        ),
        "interpretation": (
            "Values are differences in 20-day returns (stock return - sector return). "
            "Positive values (e.g., +0.05) mean stock gained 5% more than sector (outperforming). "
            "Negative values (e.g., -0.03) mean stock gained 3% less than sector (underperforming). "
            "Typical filter thresholds: > 0.02 (2% outperformance) for sector leaders, "
            "< -0.02 (-2% underperformance) for sector laggards. "
            "This feature filters out 'beta-driven' moves where the whole sector moves together."
        )
    },
    "mkt_spy_sma200_slope": {
        "name": "SPY SMA200 Slope",
        "description": (
            "SPY SMA200 slope (direction/persistence of market's long-term trend). "
            "Measures the direction and persistence of the market's long-term trend. "
            "Provides market trend direction context: uptrend vs downtrend vs flat/range regime. "
            "Percentile rank normalized over 1260 days (~5 years) and remapped to [-1, +1]."
        ),
        "interpretation": (
            "Normalized values are in [-1, +1] range. "
            "+1 indicates strong uptrend (top percentile), "
            "0 indicates neutral/flat trend (median), "
            "-1 indicates strong downtrend (bottom percentile). "
            "Typical filter thresholds: > 0.5 for strong uptrend conditions, "
            "< -0.5 for downtrend conditions. "
            "Pairs with SPY Distance from SMA200: position + direction = complete market context."
        )
    },
    
    # Candlestick Features
    "candle_body_pct": {
        "name": "Candle Body %",
        "description": (
            "Candle body percentage of total range. Measures candle strength and price action clarity. "
            "0.0 indicates no body (doji), 1.0 indicates full body (no wicks)."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates no body (doji - open equals close), "
            "1.0 indicates full body (no wicks). "
            "Typical filter thresholds: > 0.5 for strong price action (clear direction)."
        )
    },
    "candle_upper_wick_pct": {
        "name": "Candle Upper Wick %",
        "description": (
            "Upper wick percentage of total range. Measures selling pressure at highs, "
            "rejection of higher prices. Higher values indicate more rejection at highs."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates no upper wick, 1.0 indicates full upper wick (entire range is upper wick). "
            "Typical filter thresholds: > 0.3 for significant selling pressure at highs."
        )
    },
    "candle_lower_wick_pct": {
        "name": "Candle Lower Wick %",
        "description": (
            "Lower wick percentage of total range. Measures buying pressure at lows, "
            "rejection of lower prices. Higher values indicate more rejection at lows."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates no lower wick, 1.0 indicates full lower wick (entire range is lower wick). "
            "Typical filter thresholds: > 0.3 for significant buying pressure at lows."
        )
    },
    
    # Price Action Features
    "higher_high_10d": {
        "name": "Higher High (10d)",
        "description": (
            "Binary flag indicating if current close is higher than previous 10-day maximum. "
            "Indicates bullish momentum and potential trend continuation."
        ),
        "interpretation": (
            "Binary values: 0 = current close is NOT higher than previous 10-day max, "
            "1 = current close IS higher (higher high). "
            "Filter for value = 1 to identify bullish momentum."
        )
    },
    "higher_low_10d": {
        "name": "Higher Low (10d)",
        "description": (
            "Binary flag indicating if current close is higher than previous 10-day minimum. "
            "Indicates bullish momentum and potential trend continuation with higher lows."
        ),
        "interpretation": (
            "Binary values: 0 = current close is NOT higher than previous 10-day min, "
            "1 = current close IS higher (higher low). "
            "Filter for value = 1 to identify bullish momentum with higher lows."
        )
    },
    "donchian_position": {
        "name": "Donchian Position",
        "description": (
            "Position within Donchian Channel (20-period) - measures breakout structure. "
            "Measures position within established trading range. Identifies when price is near breakout levels."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates close at lowest low (at lower channel), "
            "1.0 indicates close at highest high (at upper channel), "
            "0.5 indicates close in middle of channel. "
            "Typical filter thresholds: > 0.8 for near-upper-channel (breakout potential)."
        )
    },
    "donchian_breakout": {
        "name": "Donchian Breakout",
        "description": (
            "Binary flag indicating breakout above prior 20-day high close (non-lookahead). "
            "Breakouts often precede strong trending moves. Non-lookahead implementation ensures no data leakage."
        ),
        "interpretation": (
            "Binary values: 0 = close is NOT above prior 20-day high close (no breakout), "
            "1 = close IS above prior 20-day high close (breakout detected). "
            "Filter for value = 1 to identify breakout conditions."
        )
    },
    
    # Trend Features
    "trend_residual": {
        "name": "Trend Residual",
        "description": (
            "Deviation from linear trend (noise vs trend). Measures how much price deviates from linear trend, "
            "useful for mean reversion. Negative values indicate price below trend (potential oversold)."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.2, 0.2] range. "
            "Negative values indicate price below trend (potential oversold), "
            "positive values indicate price above trend (potential overbought), "
            "near 0 indicates price follows trend closely. "
            "Typical filter thresholds: < -0.05 for oversold (below trend)."
        )
    },
    "adx14": {
        "name": "ADX (14)",
        "description": (
            "Average Directional Index (14-period) - trend strength indicator. "
            "One of the most predictive free indicators. Measures trend strength independent of direction."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range (ADX / 100). "
            "0.0 indicates no trend (ranging market), "
            "0.5 indicates moderate trend, "
            "1.0 indicates very strong trend. "
            "Typical filter thresholds: > 0.25 for trending conditions (avoid ranging markets)."
        )
    },
    "aroon_up": {
        "name": "Aroon Up",
        "description": (
            "Aroon Up (25-period) - normalized measure of days since highest high, indicating uptrend maturity. "
            "Identifies if uptrend is fresh, maturing, or exhausted. Fresh trends often continue, exhausted trends often reverse."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "1.0 indicates highest high was today (fresh uptrend), "
            "0.0 indicates highest high was 25+ days ago (exhausted uptrend). "
            "Typical filter thresholds: > 0.6 for fresh/maturing uptrends."
        )
    },
    "aroon_down": {
        "name": "Aroon Down",
        "description": (
            "Aroon Down (25-period) - normalized measure of days since lowest low, indicating downtrend maturity. "
            "Identifies if downtrend is starting or ending. Fresh downtrends often continue, exhausted downtrends often reverse."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "1.0 indicates lowest low was today (fresh downtrend), "
            "0.0 indicates lowest low was 25+ days ago (exhausted downtrend). "
            "Typical filter thresholds: < 0.4 for exhausted downtrends (reversal potential)."
        )
    },
    "aroon_oscillator": {
        "name": "Aroon Oscillator",
        "description": (
            "Aroon Oscillator (25-period) - trend dominance indicator combining Aroon Up and Aroon Down. "
            "Provides clean continuous signal for ML. Combines both Aroon lines into single 'net trend pressure' measure."
        ),
        "interpretation": (
            "Normalized values are in [0.0, 1.0] range. "
            "0.0 indicates strong downtrend dominance, "
            "0.5 indicates neutral/transition, "
            "1.0 indicates strong uptrend dominance. "
            "Typical filter thresholds: > 0.6 for uptrend dominance, < 0.4 for downtrend dominance."
        )
    },
    "fractal_dimension_index": {
        "name": "Fractal Dimension Index",
        "description": (
            "Measures how 'rough' the price path is (fractal dimension). "
            "Tells model whether it's in a trend-friendly vs whipsaw environment. "
            "Very few retail systems use it – it's a genuine edge-type feature."
        ),
        "interpretation": (
            "Normalized values are in [0, 1] range. "
            "0.0-0.4 indicates smooth, trending (trend-friendly environment), "
            "0.75-1.0 indicates choppy, mean-reverting, noisy (whipsaw environment). "
            "Typical filter thresholds: < 0.6 for trend-friendly conditions."
        )
    },
    "hurst_exponent": {
        "name": "Hurst Exponent",
        "description": (
            "Quantifies whether returns persist, mean-revert, or act like noise (R/S method). "
            "Tells model if momentum features should be trusted. Great for swing trading where persistence matters."
        ),
        "interpretation": (
            "Normalized values are in [0, 1] range. "
            "> 0.5 indicates persistent/trending (moves tend to continue), "
            "< 0.5 indicates mean-reverting (moves tend to snap back), "
            "≈ 0.5 indicates near-random walk. "
            "Typical filter thresholds: > 0.55 for trending conditions (momentum signals reliable)."
        )
    },
    "price_curvature": {
        "name": "Price Curvature",
        "description": (
            "Second derivative of trend (acceleration/deceleration). "
            "Distinguishes steady trends from accelerating/rolling-over ones. "
            "Helps the model time entries inside an already-known trend."
        ),
        "interpretation": (
            "Normalized values are clipped to [-0.05, 0.05] range. "
            "Positive values indicate trend is bending up (acceleration), "
            "negative values indicate trend is bending down (deceleration/topping), "
            "near 0 indicates linear-ish trend. "
            "Typical filter thresholds: > 0.01 for accelerating trends (bullish)."
        )
    }
}


def get_feature_description(feature_name: str) -> dict:
    """
    Get feature description by normalized column name.
    
    Args:
        feature_name: Column name from feature data (e.g., 'rsi14', 'price_vs_ma200')
        
    Returns:
        Dictionary with 'name', 'description', and 'interpretation' keys,
        or None if feature not found
    """
    # Normalize feature name (lowercase, handle variations)
    normalized = feature_name.lower().strip()
    
    # Direct lookup
    if normalized in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[normalized]
    
    # Try with common variations
    variations = [
        normalized.replace("_", ""),
        normalized.replace("_normalized", ""),
        normalized.replace("normalized_", ""),
    ]
    
    for variant in variations:
        if variant in FEATURE_DESCRIPTIONS:
            return FEATURE_DESCRIPTIONS[variant]
    
    return None


def get_feature_display_name(feature_name: str) -> str:
    """
    Get readable display name for a feature.
    
    Args:
        feature_name: Column name from feature data
        
    Returns:
        Readable name if found, otherwise returns formatted version of input
    """
    desc = get_feature_description(feature_name)
    if desc:
        return desc["name"]
    
    # Fallback: format the column name nicely
    return feature_name.replace("_", " ").title()

