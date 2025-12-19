# Complete Feature Guide

This document provides comprehensive documentation for all 60 technical indicators used in the swing trading ML pipeline. Each feature includes calculation details, normalization methods, and characteristics.

---

## Table of Contents

1. [Price Features (3)](#price-features-3)
2. [Return Features (6)](#return-features-6)
3. [52-Week Features (3)](#52-week-features-3)
4. [Moving Average Features (8)](#moving-average-features-8)
5. [Volatility Features (8)](#volatility-features-8)
6. [Volume Features (5)](#volume-features-5)
7. [Momentum Features (3)](#momentum-features-3)
8. [Market Context (2)](#market-context-2)
9. [Candlestick Features (3)](#candlestick-features-3)
10. [Price Action Features (4)](#price-action-features-4)
11. [Trend Features (8)](#trend-features-8)

---

## Price Features (3)

### 1. price
**What it represents:** Raw closing price value.

**How to Calculate:**
- `price = close`

**Normalization:** None (raw price value)

**Feature Characteristics:**
- Useful for filtering by price ranges (e.g., $1-$5, >$5, >$10)
- Base price feature for price-based filters
- Not normalized - use for filtering, not ML input

**Why it's valuable:** Essential for price-based filtering similar to Finviz filters.

---

### 2. price_log
**What it represents:** Log-transformed closing price.

**How to Calculate:**
- `price_log = ln(close)`

**Normalization:** Log transformation naturally compresses large differences

**Feature Characteristics:**
- Squashes huge differences between high and low priced stocks
- Makes price comparable across different price ranges
- Prefer this over raw price for ML applications

**Why it's valuable:** Makes price features suitable for ML models that need normalized inputs.

---

### 3. price_vs_ma200
**What it represents:** Price normalized relative to 200-day moving average.

**How to Calculate:**
1. `SMA200 = close.rolling(200).mean()`
2. `price_vs_ma200 = close / SMA200`

**Normalization:** Ratio normalization (no clipping)

**Feature Characteristics:**
- Values > 1.0: Price above long-term average (bullish)
- Values < 1.0: Price below long-term average (bearish)
- Normalizes price relative to long-term baseline
- Comparable across different price ranges

**Why it's valuable:** Provides long-term context for current price level.

---

## Return Features (6)

### 4. daily_return
**What it represents:** Daily percentage return.

**How to Calculate:**
1. `daily_return = (close / close.shift(1) - 1)`
2. Clip to `[-0.2, 0.2]` (±20%)

**Normalization:** Clipped to ±20% to cap extreme moves

**Feature Characteristics:**
- Range: [-0.2, 0.2] (±20%)
- Prevents outliers from dominating the feature
- Suitable for ML models

**Why it's valuable:** Captures daily momentum and price changes.

---

### 5. gap_pct
**What it represents:** Gap percentage between today's open and yesterday's close.

**How to Calculate:**
1. `gap_pct = (open - close.shift(1)) / close.shift(1)`
2. Clip to `[-0.2, 0.2]` (±20%)

**Normalization:** Clipped to ±20% to cap extreme gaps

**Feature Characteristics:**
- Positive values: Gap-up (bullish)
- Negative values: Gap-down (bearish)
- Range: [-0.2, 0.2] (±20%)
- Prevents extreme gap outliers

**Why it's valuable:** Measures overnight momentum and gap trading opportunities.

---

### 6. weekly_return_5d
**What it represents:** 5-day (weekly) percentage return.

**How to Calculate:**
1. `weekly_return_5d = close.pct_change(5)`
2. Clip to `[-0.3, 0.3]` (±30%)

**Normalization:** Clipped to ±30% to cap extreme moves

**Feature Characteristics:**
- Range: [-0.3, 0.3] (±30%)
- Measures weekly momentum
- Prevents outliers from dominating

**Why it's valuable:** Captures short-term momentum over approximately one week.

---

### 7. monthly_return_21d
**What it represents:** 21-day (monthly) percentage return.

**How to Calculate:**
1. `monthly_return_21d = close.pct_change(21)`
2. Clip to `[-0.5, 0.5]` (±50%)

**Normalization:** Clipped to ±50% to cap extreme moves

**Feature Characteristics:**
- Range: [-0.5, 0.5] (±50%)
- Measures monthly momentum
- Prevents outliers from dominating

**Why it's valuable:** Captures medium-term momentum over approximately one month.

---

### 8. quarterly_return_63d
**What it represents:** 63-day (quarterly) percentage return.

**How to Calculate:**
1. `quarterly_return_63d = close.pct_change(63)`
2. Clip to `[-1.0, 1.0]` (±100%)

**Normalization:** Clipped to ±100% to cap extreme moves

**Feature Characteristics:**
- Range: [-1.0, 1.0] (±100%)
- Measures quarterly momentum
- Prevents outliers from dominating

**Why it's valuable:** Captures longer-term momentum over approximately one quarter.

---

### 9. ytd_return
**What it represents:** Year-to-Date percentage return.

**How to Calculate:**
1. `first_close_of_year = close.groupby(year).transform('first')`
2. `ytd_return = (close / first_close_of_year) - 1`
3. Clip to `[-1.0, 2.0]`

**Normalization:** Clipped to [-1.0, 2.0] (minimum -100%, maximum +200%)

**Feature Characteristics:**
- Range: [-1.0, 2.0] (-100% to +200%)
- Measures performance from start of year
- Prevents extreme outliers

**Why it's valuable:** Provides year-to-date performance context.

---

## 52-Week Features (3)

### 10. dist_52w_high
**What it represents:** Distance from 52-week high.

**How to Calculate:**
1. `high_52w = close.rolling(252).max()`
2. `dist_52w_high = (close / high_52w) - 1`
3. Clip to `[-1.0, 0.5]`

**Normalization:** Clipped to [-1.0, 0.5]

**Feature Characteristics:**
- 0.0: At 52-week high
- Negative: Below 52-week high (more negative = further below)
- Positive: Above 52-week high (rare, new highs)
- Range: [-1.0, 0.5]

**Why it's valuable:** Measures how far price is from recent highs, useful for breakout detection.

---

### 11. dist_52w_low
**What it represents:** Distance from 52-week low.

**How to Calculate:**
1. `low_52w = close.rolling(252).min()`
2. `dist_52w_low = (close / low_52w) - 1`
3. Clip to `[-0.5, 2.0]`

**Normalization:** Clipped to [-0.5, 2.0]

**Feature Characteristics:**
- 0.0: At 52-week low
- Positive: Above 52-week low (more positive = further above)
- Negative: Below 52-week low (rare, new lows)
- Range: [-0.5, 2.0]

**Why it's valuable:** Measures how far price is from recent lows, useful for support detection.

---

### 12. pos_52w
**What it represents:** Position within 52-week range (0=low, 1=high).

**How to Calculate:**
1. `high_52w = close.rolling(252).max()`
2. `low_52w = close.rolling(252).min()`
3. `pos_52w = (close - low_52w) / (high_52w - low_52w)`
4. Clip to `[0.0, 1.0]`

**Normalization:** Already normalized to [0, 1] range

**Feature Characteristics:**
- 0.0: At 52-week low
- 1.0: At 52-week high
- 0.5: At midpoint of 52-week range
- Range: [0.0, 1.0]

**Why it's valuable:** Normalized position within yearly range, useful for relative strength analysis.

---

## Moving Average Features (8)

### 13. sma20_ratio
**What it represents:** Price relative to 20-day SMA.

**How to Calculate:**
1. `SMA20 = close.rolling(20).mean()`
2. `sma20_ratio = close / SMA20`
3. Clip to `[0.5, 1.5]`

**Normalization:** Clipped to [0.5, 1.5]

**Feature Characteristics:**
- 1.0: Price equals SMA20
- > 1.0: Price above SMA20 (bullish)
- < 1.0: Price below SMA20 (bearish)
- Range: [0.5, 1.5]

**Why it's valuable:** Short-term trend indicator relative to 20-day average.

---

### 14. sma50_ratio
**What it represents:** Price relative to 50-day SMA.

**How to Calculate:**
1. `SMA50 = close.rolling(50).mean()`
2. `sma50_ratio = close / SMA50`
3. Clip to `[0.5, 1.5]`

**Normalization:** Clipped to [0.5, 1.5]

**Feature Characteristics:**
- 1.0: Price equals SMA50
- > 1.0: Price above SMA50 (bullish)
- < 1.0: Price below SMA50 (bearish)
- Range: [0.5, 1.5]

**Why it's valuable:** Medium-term trend indicator relative to 50-day average.

---

### 15. sma200_ratio
**What it represents:** Price relative to 200-day SMA.

**How to Calculate:**
1. `SMA200 = close.rolling(200).mean()`
2. `sma200_ratio = close / SMA200`
3. Clip to `[0.5, 2.0]`

**Normalization:** Clipped to [0.5, 2.0]

**Feature Characteristics:**
- 1.0: Price equals SMA200
- > 1.0: Price above SMA200 (bullish, long-term uptrend)
- < 1.0: Price below SMA200 (bearish, long-term downtrend)
- Range: [0.5, 2.0]

**Why it's valuable:** Long-term trend indicator, classic bull/bear market signal.

---

### 16. sma20_sma50_ratio
**What it represents:** Ratio of SMA20 to SMA50 (moving average crossover).

**How to Calculate:**
1. `SMA20 = close.rolling(20).mean()`
2. `SMA50 = close.rolling(50).mean()`
3. `sma20_sma50_ratio = SMA20 / SMA50`
4. Clip to `[0.8, 1.2]`

**Normalization:** Clipped to [0.8, 1.2]

**Feature Characteristics:**
- 1.0: SMA20 equals SMA50 (neutral)
- > 1.0: SMA20 above SMA50 (bullish crossover, uptrend)
- < 1.0: SMA20 below SMA50 (bearish crossover, downtrend)
- Range: [0.8, 1.2]

**Why it's valuable:** Moving average crossover indicator for short-to-medium term trends.

---

### 17. sma50_sma200_ratio
**What it represents:** Ratio of SMA50 to SMA200 (Golden Cross/Death Cross).

**How to Calculate:**
1. `SMA50 = close.rolling(50).mean()`
2. `SMA200 = close.rolling(200).mean()`
3. `sma50_sma200_ratio = SMA50 / SMA200`
4. Clip to `[0.6, 1.4]`

**Normalization:** Clipped to [0.6, 1.4]

**Feature Characteristics:**
- 1.0: SMA50 equals SMA200 (neutral)
- > 1.0: SMA50 above SMA200 (Golden Cross, bullish long-term trend)
- < 1.0: SMA50 below SMA200 (Death Cross, bearish long-term trend)
- Range: [0.6, 1.4]

**Why it's valuable:** Classic moving average crossover indicator for long-term trends.

---

### 18. sma50_slope
**What it represents:** 5-day change in SMA50, normalized by price.

**How to Calculate:**
1. `SMA50 = close.rolling(50).mean()`
2. `sma50_slope = SMA50.diff(5) / close`
3. Clip to `[-0.1, 0.1]`

**Normalization:** Clipped to [-0.1, 0.1]

**Feature Characteristics:**
- 0.0: SMA50 is flat (no change)
- > 0.0: SMA50 is rising (bullish momentum)
- < 0.0: SMA50 is falling (bearish momentum)
- Range: [-0.1, 0.1]

**Why it's valuable:** Measures rate of change (slope) of medium-term trend.

---

### 19. sma200_slope
**What it represents:** 10-day change in SMA200, normalized by price.

**How to Calculate:**
1. `SMA200 = close.rolling(200).mean()`
2. `sma200_slope = SMA200.diff(10) / close`
3. Clip to `[-0.1, 0.1]`

**Normalization:** Clipped to [-0.1, 0.1]

**Feature Characteristics:**
- 0.0: SMA200 is flat (no change)
- > 0.0: SMA200 is rising (bullish long-term momentum)
- < 0.0: SMA200 is falling (bearish long-term momentum)
- Range: [-0.1, 0.1]

**Why it's valuable:** Measures rate of change (slope) of long-term trend.

---

### 20. kama_slope
**What it represents:** KAMA (Kaufman Adaptive Moving Average) Slope - adaptive trend strength indicator.

**How to Calculate:**
1. Calculate Efficiency Ratio (ER):
   - `change = abs(close - close[10 periods ago])`
   - `volatility = sum(abs(close.diff()) for 10 periods)`
   - `ER = change / volatility` (clipped to [0, 1])
2. Calculate Smoothing Constant (SC):
   - `fast_SC = 2 / (2 + 1) = 0.667`
   - `slow_SC = 2 / (30 + 1) = 0.065`
   - `SC = [ER * (fast_SC - slow_SC) + slow_SC]^2`
3. Calculate KAMA:
   - Initialize with SMA(10) for first period
   - `KAMA[t] = KAMA[t-1] + SC[t] * (close[t] - KAMA[t-1])`
4. Calculate KAMA Slope:
   - `kama_slope = kama.diff() / close`
   - Clip to `[-0.1, 0.1]`

**Normalization:** Normalized by price (kama.diff() / close), clipped to [-0.1, 0.1]

**Feature Characteristics:**
- 0.0: KAMA is flat (no change)
- > 0.0: KAMA is rising (adaptive bullish momentum)
- < 0.0: KAMA is falling (adaptive bearish momentum)
- Range: [-0.1, 0.1]
- Adapts to market efficiency: fast in trends, slow in noise
- More responsive than SMA in efficient markets, less whipsaw in choppy markets

**Why it's valuable:**
- SMA slopes measure linear trend
- KAMA slope measures adaptive trend strength
- Works better in choppy tickers
- Adapts to price smoothness (flat & slow when noisy, fast & responsive when efficient)
- Provides dynamic trend strength that changes based on volatility efficiency

---

## Volatility Features (8)

### 20. volatility_5d
**What it represents:** 5-day rolling standard deviation of daily returns.

**How to Calculate:**
1. `returns = close.pct_change()`
2. `volatility_5d = returns.rolling(5).std()`
3. Clip to `[0.0, 0.15]`

**Normalization:** Clipped to [0.0, 0.15] (0% to 15% daily volatility)

**Feature Characteristics:**
- Higher values: More volatile price movements
- Range: [0.0, 0.15] (0% to 15% daily volatility)
- Short-term volatility measure

**Why it's valuable:** Captures short-term price volatility and risk.

---

### 21. volatility_21d
**What it represents:** 21-day rolling standard deviation of daily returns.

**How to Calculate:**
1. `returns = close.pct_change()`
2. `volatility_21d = returns.rolling(21).std()`
3. Clip to `[0.0, 0.15]`

**Normalization:** Clipped to [0.0, 0.15] (0% to 15% daily volatility)

**Feature Characteristics:**
- Higher values: More volatile price movements
- Range: [0.0, 0.15] (0% to 15% daily volatility)
- Medium-term volatility measure

**Why it's valuable:** Captures medium-term price volatility and risk.

---

### 22. volatility_ratio
**What it represents:** Volatility Ratio - short-term volatility (5-day) vs long-term volatility (21-day).

**How to Calculate:**
1. `vol5 = volatility_5d` (already computed)
2. `vol21 = volatility_21d` (already computed)
3. `volatility_ratio = vol5 / vol21`
4. Clip to `[0, 2]`

**Normalization:** Clipped to [0, 2] range

**Feature Characteristics:**
- > 1: Volatility expanding (short-term vol > long-term vol)
- < 1: Volatility contracting (short-term vol < long-term vol)
- ≈ 1: Stable regime (short-term vol ≈ long-term vol)
- Range: [0, 2]
- Identifies volatility expansion/compression regimes
- More cleanly identifies volatility expansion/compression than ATR alone

**Why it's valuable:**
- Captures regime shifts
- Improves breakout & pullback predictions
- Helps differentiate choppy vs trending markets
- More cleanly identifies volatility expansion/compression than ATR alone
- Extremely important for predicting swing duration and follow-through
- The ratio of short-term volatility to long-term volatility identifies volatility expansion/compression regimes

---

### 23. atr14_normalized
**What it represents:** Average True Range (14-day) normalized by price.

**How to Calculate:**
1. `TR = max(high-low, abs(high-prev_close), abs(low-prev_close))`
2. `ATR14 = TR.rolling(14).mean()`
3. `atr14_normalized = ATR14 / close`
4. Clip to `[0.0, 0.2]`

**Normalization:** Clipped to [0.0, 0.2] (0% to 20% of price)

**Feature Characteristics:**
- Measures volatility based on true range
- Range: [0.0, 0.2] (0% to 20% of price)
- Accounts for gaps in price movement

**Why it's valuable:** More accurate volatility measure that accounts for gaps and overnight moves.

---

### 24. bollinger_band_width
**What it represents:** Bollinger Band Width (log normalized) - measures volatility compression/expansion.

**How to Calculate:**
1. `mid = SMA(close, 20)`
2. `std = close.rolling(20).std()`
3. `upper = mid + 2 * std`
4. `lower = mid - 2 * std`
5. `bbw = (upper - lower) / mid`
6. `bollinger_band_width = log1p(bbw)`

**Normalization:** Log normalization using log1p

**Feature Characteristics:**
- Captures volatility squeezes and expansions
- Predicts breakout probability
- Identifies trend exhaustion
- Detects range tightening
- Signals market regime shifts
- BB squeezes often precede strong 10-30 day swings

**Why it's valuable:** One of the best predictors of breakouts and volatility expansions, perfect for swing trading.

---

### 25. ttm_squeeze_on
**What it represents:** TTM Squeeze condition - binary flag indicating volatility contraction (squeeze).

**How to Calculate:**
1. Bollinger Bands:
   - `mid = SMA(close, 20)`
   - `std = close.rolling(20).std()`
   - `upper_bb = mid + 2*std`
   - `lower_bb = mid - 2*std`
2. Keltner Channels:
   - `atr = ATR(20)` (calculated from True Range)
   - `upper_kc = mid + 1.5*atr`
   - `lower_kc = mid - 1.5*atr`
3. Squeeze condition:
   - `squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)`

**Normalization:** Binary (0 or 1)

**Feature Characteristics:**
- 0: No squeeze (Bollinger Bands wider than Keltner Channels)
- 1: Squeeze active (Bollinger Bands inside Keltner Channels - volatility contraction)
- Binary flag
- Identifies tight trading ranges that often precede breakouts

**Why it's valuable:**
- This is the #1 breakout indicator used by quant retail traders
- Catches explosive moves after volatility compression
- Identifies tight trading ranges that often precede breakouts
- Combines volatility analysis with momentum direction

---

### 26. ttm_squeeze_momentum
**What it represents:** TTM Squeeze momentum (normalized) - momentum direction during squeeze.

**How to Calculate:**
1. `mid = SMA(close, 20)`
2. `squeeze_momentum = close - mid`
3. `squeeze_momentum_norm = squeeze_momentum / close`

**Normalization:** Normalized by price (momentum / close)

**Feature Characteristics:**
- Positive: Price above SMA20 (bullish momentum)
- Negative: Price below SMA20 (bearish momentum)
- Near 0: Price at SMA20 (neutral)
- Normalized by price makes it comparable across different price ranges

**Why it's valuable:**
- Provides momentum direction during squeeze conditions
- Helps identify which direction the breakout is likely to occur
- Normalized by price makes it comparable across different price ranges
- Works in combination with squeeze_on to identify high-probability setups

---

### 27. volatility_of_volatility
**What it represents:** Measures how unstable volatility itself is (meta volatility).

**How to Calculate:**
1. Compute 21-day volatility: `σ_21,t = std(r_{t-20}, ..., r_t)`
2. Compute rolling std of volatility over 21 bars: `VoV_t = std(σ_21,{t-20}, ..., σ_21,t)`
3. Normalize by dividing by long-term average volatility: `VoV_rel = VoV_t / mean(σ_21)`
4. Clip to `[0, 3]`

**Normalization:**
- Division by long-term average volatility makes it relative and comparable
- Clipped to [0, 3] to limit extreme values

**Feature Characteristics:**
- Low VoV → calm, stable regime (signals behave more cleanly)
- High VoV → chaotic regime, risk of whipsaws/gaps/wild moves
- Range: [0, 3] (normalized relative to long-term average volatility)
- Tells model whether volatility indicators are reliable or chaotic

**Why it's valuable:**
- Tells the model whether volatility indicators are reliable or chaotic
- Helps risk-aware decision making (e.g., avoid super unstable regimes)
- Strong context feature when paired with ATR, BB width, TTM squeeze, volatility_ratio

---

## Volume Features (5)

### 28. log_volume
**What it represents:** Log-transformed volume.

**How to Calculate:**
- `log_volume = log1p(volume)`

**Normalization:** Log transformation (log1p) naturally compresses large values

**Feature Characteristics:**
- Handles zero volumes gracefully
- Compresses wide range of volume values
- Makes volume comparable across stocks

**Why it's valuable:** Normalizes volume for ML models, handles wide range of volume values.

---

### 28. log_avg_volume_20d
**What it represents:** Log-transformed 20-day average volume.

**How to Calculate:**
1. `vol_avg20 = volume.rolling(20).mean()`
2. `log_avg_volume_20d = log1p(vol_avg20)`

**Normalization:** Log transformation (log1p) naturally compresses large values

**Feature Characteristics:**
- Smoothed view of volume trends
- Medium-term volume baseline
- Normalized for ML models

**Why it's valuable:** Provides smoothed, normalized view of volume trends.

---

### 29. relative_volume
**What it represents:** Current volume relative to 20-day average.

**How to Calculate:**
1. `vol_avg20 = volume.rolling(20).mean()`
2. `rvol = volume / vol_avg20`
3. Clip to `[0, 10]`
4. `relative_volume = log1p(rvol)`

**Normalization:** Clipped to [0, 10], then log1p transformation

**Feature Characteristics:**
- > 1.0: Above-average volume
- < 1.0: Below-average volume
- Log-normalized for ML models

**Why it's valuable:** Identifies unusual volume activity, important for breakout detection.

---

### 30. chaikin_money_flow
**What it represents:** Chaikin Money Flow (20-period) - accumulation vs distribution.

**How to Calculate:**
1. `mfm = (2*close - high - low) / (high - low)` (Money Flow Multiplier)
2. `mfv = mfm * volume` (Money Flow Volume)
3. `cmf = sum(mfv over 20d) / sum(volume over 20d)`
4. Clip to `[-1, 1]`

**Normalization:** Already normalized to [-1, 1] range

**Feature Characteristics:**
- -1.0: Strong selling pressure / distribution
- 0.0: Neutral (balanced buying/selling)
- +1.0: Strong buying pressure / accumulation
- Combines price action with volume
- Range: [-1.0, 1.0]

**Why it's valuable:**
- Exposes quiet accumulation before breakouts
- Combines price action with volume (stronger signal than price alone)
- Detects institutional accumulation/distribution
- One of the strongest predictors of future swings

---

### 31. obv_momentum
**What it represents:** OBV Momentum (OBV Rate of Change) - 10-day percentage change of On-Balance Volume.

**How to Calculate:**
1. Build OBV (On-Balance Volume):
   - Close up → add volume
   - Close down → subtract volume
   - Equal → no change
2. Calculate 10-day percentage change of OBV:
   - `obv_roc = OBV.pct_change(10)`
3. Normalize by clipping to `[-0.5, 0.5]`

**Normalization:** Clipped to [-0.5, 0.5] (±50% change)

**Feature Characteristics:**
- Positive: Volume accelerating upward (bullish)
- Negative: Volume accelerating downward (bearish)
- Near 0: Volume momentum neutral
- Range: [-0.5, 0.5] (±50% change)
- Gives volume acceleration, not just volume level

**Why it's valuable:**
- Gives volume acceleration, not just volume level
- Works extremely well with breakouts and volatility squeezes
- One of the highest-impact free indicators you can add
- Catches institutional volume movements before price shifts
- Big funds often move volume before price shifts — OBV ROC catches that

---

## Momentum Features (9)

### 32. rsi14
**What it represents:** Relative Strength Index (14-period) with centered normalization.

**How to Calculate:**
1. `delta = close.diff()`
2. `gain = delta.clip(lower=0)`
3. `loss = -delta.clip(upper=0)`
4. `avg_gain = gain.rolling(14).mean()`
5. `avg_loss = loss.rolling(14).mean()`
6. `rs = avg_gain / avg_loss`
7. `rsi = 100 - (100 / (1 + rs))`
8. `rsi14 = (rsi - 50) / 50` (centered normalization)

**Normalization:** Centered to [-1, +1] range: (rsi - 50) / 50

**Feature Characteristics:**
- -1.0: RSI = 0 (extremely oversold)
- 0.0: RSI = 50 (neutral)
- +1.0: RSI = 100 (extremely overbought)
- Range: [-1.0, 1.0]

**Why it's valuable:** Classic momentum oscillator for overbought/oversold conditions.

---

### 33. macd_histogram_normalized
**What it represents:** MACD Histogram normalized by price - measures momentum acceleration/deceleration.

**How to Calculate:**
1. `EMA12 = close.ewm(span=12).mean()`
2. `EMA26 = close.ewm(span=26).mean()`
3. `macd_line = EMA12 - EMA26`
4. `signal_line = macd_line.ewm(span=9).mean()`
5. `macd_hist = macd_line - signal_line`
6. `macd_histogram_normalized = macd_hist / close`

**Normalization:** Normalized by price (histogram / close)

**Feature Characteristics:**
- Most predictive part of MACD
- Shows strength of trend
- Detects early momentum shifts
- Captures trend momentum
- Detects divergence
- Identifies turning points
- Shows acceleration vs deceleration
- Much more expressive than RSI alone

**Why it's valuable:** The histogram is the most predictive part of MACD, detecting early momentum shifts and trend strength.

---

### 34. ppo_histogram
**What it represents:** PPO (Percentage Price Oscillator) Histogram - percentage-based momentum acceleration/deceleration.

**How to Calculate:**
1. Calculate two EMAs:
   - `ema12 = EMA(close, 12)`
   - `ema26 = EMA(close, 26)`
2. Calculate PPO line:
   - `ppo = (ema12 - ema26) / ema26`
3. Calculate PPO signal:
   - `ppo_signal = EMA(ppo, 9)`
4. Calculate PPO histogram:
   - `ppo_hist = ppo - ppo_signal`
5. Normalize by clipping to `[-0.2, 0.2]`

**Normalization:** Clipped to [-0.2, 0.2] (typically ranges around [-0.1, 0.1])

**Feature Characteristics:**
- Positive: PPO accelerating above signal (bullish momentum acceleration)
- Negative: PPO decelerating below signal (bearish momentum deceleration)
- Near 0: PPO and signal converging (momentum neutral)
- Range: [-0.2, 0.2]
- Scale-invariant and cross-ticker comparable
- Measures momentum in percent (unlike MACD which is in absolute price units)

**Why it's valuable:**
- Less redundant with MACD than you'd think (percentage vs absolute)
- Captures percent-based momentum cleanly
- Often improves ML performance with multi-ticker training
- More stable across expensive vs cheap stocks
- Cross-ticker comparable (unlike MACD which is in price units)
- PPO measures momentum in percent, making it scale-invariant

---

### 35. dpo
**What it represents:** DPO (Detrended Price Oscillator, 20-period) - cyclical indicator that removes long-term trend.

**How to Calculate:**
1. Calculate centered SMA:
   - `period = 20`
   - `sma = close.rolling(20).mean()`
   - `shifted_sma = sma.shift(20//2 + 1)` (shift by 11 periods)
2. Calculate DPO:
   - `dpo = close - shifted_sma`
3. Normalize by closing price:
   - `dpo_norm = dpo / close`
4. Clip to `[-0.2, 0.2]`

**Normalization:** Normalized by closing price (dpo / close), clipped to [-0.2, 0.2]

**Feature Characteristics:**
- Positive: Price above detrended average (cycle peak, overextended)
- Negative: Price below detrended average (cycle trough, compressed)
- Near 0: Price at detrended average (neutral)
- Range: [-0.2, 0.2]
- Highlights short-term price cycles
- Removes long-term trend to focus on cyclical patterns

**Why it's valuable:**
- Gives the model cycle structure, which no other feature gives
- Helps detect "overextended" or "compressed" prices
- Complements CCI & Williams %R
- Removes long-term trend to focus on cyclical patterns
- Perfect for identifying mean-reversion zones
- Helps detect cycle peaks, cycle troughs, trend pullbacks, and mean-reversion zones
- Perfect for 10-30 day windows

---

### 36. roc10
**What it represents:** ROC (Rate of Change) 10-period - short-term momentum velocity indicator.

**How to Calculate:**
1. `roc10 = (close - close.shift(10)) / close.shift(10)`
2. Clip to `[-0.5, 0.5]`

**Normalization:** Clipped to [-0.5, 0.5] (±50% change over 10 periods)

**Feature Characteristics:**
- Positive: Price rising over 10 periods (bullish momentum)
- Negative: Price falling over 10 periods (bearish momentum)
- Near 0: Price relatively stable (neutral momentum)
- Range: [-0.5, 0.5]
- Standardized and directional momentum indicator
- Captures velocity, not just simple percent change

**Why it's valuable:**
- Highly predictive in breakouts and pullbacks
- A more expressive form of momentum than basic log returns
- ROC10 + ROC20 = excellent short/medium-term momentum pair
- Captures velocity, not just simple percent change
- Standardized momentum indicator (unlike returns)
- Different from log returns because ROC captures velocity

---

### 37. roc20
**What it represents:** ROC (Rate of Change) 20-period - medium-term momentum velocity indicator.

**How to Calculate:**
1. `roc20 = (close - close.shift(20)) / close.shift(20)`
2. Clip to `[-0.7, 0.7]`

**Normalization:** Clipped to [-0.7, 0.7] (±70% change over 20 periods)

**Feature Characteristics:**
- Positive: Price rising over 20 periods (bullish momentum)
- Negative: Price falling over 20 periods (bearish momentum)
- Near 0: Price relatively stable (neutral momentum)
- Range: [-0.7, 0.7]
- Standardized and directional momentum indicator
- Captures velocity, not just simple percent change

**Why it's valuable:**
- Highly predictive in breakouts and pullbacks
- A more expressive form of momentum than basic log returns
- ROC10 + ROC20 = excellent short/medium-term momentum pair
- Captures velocity, not just simple percent change
- Standardized momentum indicator (unlike returns)
- Different from log returns because ROC captures velocity

---

### 38. stochastic_k14
**What it represents:** Stochastic Oscillator %K (14-period) - position within trading range.

**How to Calculate:**
1. `low_14 = low.rolling(14).min()`
2. `high_14 = high.rolling(14).max()`
3. `stochastic_k14 = (close - low_14) / (high_14 - low_14)`
4. Clip to `[0, 1]`

**Normalization:** Already normalized to [0, 1] range

**Feature Characteristics:**
- 0.0: Close at lowest low (extremely oversold)
- 0.5: Close in middle of range (neutral)
- 1.0: Close at highest high (extremely overbought)
- Range: [0.0, 1.0]
- Better than RSI in many trend scenarios
- Captures range compression & exhaustion
- Helps detect early reversals & continuation setups

**Why it's valuable:** Directly captures overbought/oversold relative to range, better than RSI in many scenarios.

---

### 39. cci20
**What it represents:** CCI (Commodity Channel Index, 20-period) - standardized distance from trend oscillator.

**How to Calculate:**
1. `typical_price = (high + low + close) / 3`
2. `sma_tp = typical_price.rolling(20).mean()`
3. `mean_deviation = mean(abs(typical_price - sma_tp) over 20d)`
4. `cci = (typical_price - sma_tp) / (0.015 * mean_deviation)`
5. `cci20 = tanh(cci / 100)`

**Normalization:** Normalized using tanh compression: `tanh(cci / 100)`

**Feature Characteristics:**
- High CCI (>100): momentum burst, overbought
- Low CCI (<-100): selling pressure, oversold
- Near 0: price near typical mean
- Range: approximately [-1, 1] after tanh, but typically [-0.76, 0.76] for CCI in [-100, 100]
- Hybrid indicator combining momentum, volatility, and mean reversion
- Measures how far price deviates from its typical mean relative to volatility

**Why it's valuable:**
- Model lacks a standardized "distance from trend" oscillator
- CCI captures trend exhaustion & reversion points
- Great for swing trading windows
- Adds information RSI & Stochastic do NOT cover
- Hybrid indicator combining momentum, volatility, and mean reversion
- Kind of a hybrid between RSI, momentum, and volatility

---

### 40. williams_r14
**What it represents:** Williams %R (14-period) - range momentum/reversion oscillator.

**How to Calculate:**
1. `highest_high = high.rolling(14).max()`
2. `lowest_low = low.rolling(14).min()`
3. `williams_r = (highest_high - close) / (highest_high - lowest_low) * -100`
4. Normalize: `williams_r_norm = -(williams_r / 100)`
5. Clip to `[0, 1]`

**Normalization:** Normalized from [-100, 0] to [0, 1] range: `-(williams_r / 100)`

**Feature Characteristics:**
- 0.0: Close at highest high (extremely overbought)
- 0.5: Close in middle of range (neutral)
- 1.0: Close at lowest low (extremely oversold)
- Range: [0.0, 1.0]
- Very sensitive to reversal points
- Where Stochastic goes 0 → 1, %R goes -100 → 0 (inverted scale)

**Why it's valuable:**
- Very sensitive to reversal points
- Strong complement to RSI and Stochastic
- Helps catch swing entries inside trends
- Detects pullbacks within trends, oversold bounces, momentum shifts in range markets
- Provides "pressure" version of range position (complement to Stochastic %K)
- Model has Stochastic %K (range position) but not the "pressure" version - Williams %R fills that gap

---

## Market Context (2)

### 41. beta_spy_252d
**What it represents:** Rolling beta vs SPY over 252 trading days.

**How to Calculate:**
1. `stock_ret = log(close / close.shift(1))`
2. `spy_ret = log(spy_close / spy_close.shift(1))`
3. Calculate rolling covariance and variance over 252 days
4. `beta = cov / var`
5. `beta_spy_252d = ((beta + 1) / 4).clip(0, 1)`

**Normalization:** Normalized to [0, 1] range: ((beta + 1) / 4).clip(0, 1)

**Feature Characteristics:**
- Measures stock's sensitivity to market movements
- 0.0: Beta = -1 (inverse correlation)
- 0.25: Beta = 0 (no correlation)
- 0.5: Beta = 1 (moves with market)
- 1.0: Beta = 3 (highly sensitive to market)
- Range: [0.0, 1.0]

**Why it's valuable:** Provides market context - how stock moves relative to overall market.

---

### 42. mkt_spy_dist_sma200
**What it represents:** SPY distance from SMA200 (market extension vs long-term trend).

**How to Calculate:**
1. Calculate SPY SMA200: `spy_sma200 = mean(SPY_close over 200 trading days)`
2. Calculate distance: `dist = (SPY_close / spy_sma200) - 1`
3. Calculate rolling z-score: `z = (dist - mean_rolling(dist)) / std_rolling(dist)`
   - Rolling window: 1260 days (~5 years)
4. Clip to `[-3, 3]`

**Normalization:**
- Z-score normalization makes it comparable across different market regimes
- Clipping to [-3, 3] limits extreme values
- Range: [-3, 3] (standard deviations from mean)

**Feature Characteristics:**
- Higher (positive) = more risk-on / bullish environment (market extended above trend)
- Near 0 = neutral (market at trend baseline)
- Lower (negative) = risk-off / bearish regime (market below trend)
- Range: [-3, 3] (standard deviations from mean)
- First 1260 days will have NaN values (insufficient data for z-score)

**Why it's valuable:**
- Provides market regime context (risk-on vs risk-off)
- Helps model understand market environment (extended vs mean-reverting)
- Complements beta_spy_252d (correlation) with extension/regime information
- Critical for swing trading where market regime matters
- Different from beta: measures market extension, not stock correlation

---

## Candlestick Features (3)

### 43. candle_body_pct
**What it represents:** Candle body percentage of total range.

**How to Calculate:**
1. `body = abs(close - open)`
2. `range = high - low`
3. `candle_body_pct = body / range`
4. Clip to `[0, 1]` (handles division by zero)

**Normalization:** Already normalized to [0, 1] range

**Feature Characteristics:**
- 0.0: No body (doji - open equals close)
- 1.0: Full body (no wicks - body equals range)
- Range: [0.0, 1.0]

**Why it's valuable:** Measures candle strength and price action clarity.

---

### 44. candle_upper_wick_pct
**What it represents:** Upper wick percentage of total range.

**How to Calculate:**
1. `upper_wick = high - max(close, open)`
2. `range = high - low`
3. `candle_upper_wick_pct = upper_wick / range`
4. Clip to `[0, 1]` (handles division by zero)

**Normalization:** Already normalized to [0, 1] range

**Feature Characteristics:**
- 0.0: No upper wick
- 1.0: Full upper wick (entire range is upper wick)
- Range: [0.0, 1.0]

**Why it's valuable:** Measures selling pressure at highs, rejection of higher prices.

---

### 45. candle_lower_wick_pct
**What it represents:** Lower wick percentage of total range.

**How to Calculate:**
1. `lower_wick = min(close, open) - low`
2. `range = high - low`
3. `candle_lower_wick_pct = lower_wick / range`
4. Clip to `[0, 1]` (handles division by zero)

**Normalization:** Already normalized to [0, 1] range

**Feature Characteristics:**
- 0.0: No lower wick
- 1.0: Full lower wick (entire range is lower wick)
- Range: [0.0, 1.0]

**Why it's valuable:** Measures buying pressure at lows, rejection of lower prices.

---

## Price Action Features (4)

### 46. higher_high_10d
**What it represents:** Binary flag indicating if current close is higher than previous 10-day maximum.

**How to Calculate:**
1. `prev_10d_max = close.shift(1).rolling(10).max()`
2. `higher_high_10d = (close > prev_10d_max).astype(int)`

**Normalization:** Binary (0 or 1)

**Feature Characteristics:**
- 0: Current close is NOT higher than previous 10-day max
- 1: Current close IS higher than previous 10-day max (higher high)
- Binary flag

**Why it's valuable:** Indicates bullish momentum and potential trend continuation.

---

### 47. higher_low_10d
**What it represents:** Binary flag indicating if current close is higher than previous 10-day minimum.

**How to Calculate:**
1. `prev_10d_min = close.shift(1).rolling(10).min()`
2. `higher_low_10d = (close > prev_10d_min).astype(int)`

**Normalization:** Binary (0 or 1)

**Feature Characteristics:**
- 0: Current close is NOT higher than previous 10-day min
- 1: Current close IS higher than previous 10-day min (higher low)
- Binary flag

**Why it's valuable:** Indicates bullish momentum and potential trend continuation with higher lows.

---

### 48. swing_low_10d
**What it represents:** Recent swing low (10-day) - the lowest low price over the last 10 days.

**How to Calculate:**
1. `swing_low_10d = low.rolling(10, min_periods=1).min()`

**Normalization:** None (raw price value)

**Feature Characteristics:**
- Returns the actual swing low price (not normalized)
- Uses the 'low' price (not close) to capture the actual swing low point
- Lower values indicate stronger support levels
- Used in conjunction with entry price to calculate stop distance
- Identifies the most recent structural support level

**Why it's valuable:**
- Identifies structural support levels for stop-loss placement
- Helps determine risk management (distance from entry to swing low)
- Provides context for price action analysis
- Used for calculating stop-loss distances in trading strategies

---

### 49. donchian_position
**What it represents:** Position within Donchian Channel (20-period) - measures breakout structure.

**How to Calculate:**
1. `donchian_high_20 = high.rolling(20).max()`
2. `donchian_low_20 = low.rolling(20).min()`
3. `donchian_position = (close - donchian_low_20) / (donchian_high_20 - donchian_low_20)`
4. Clip to `[0, 1]`

**Normalization:** Already normalized to [0, 1] range

**Feature Characteristics:**
- 0.0: Close at the lowest low (at lower channel)
- 0.5: Close in the middle of the channel
- 1.0: Close at the highest high (at upper channel)
- Values > 1.0: Breakout above upper channel (clipped to 1.0)
- Values < 0.0: Breakdown below lower channel (clipped to 0.0)
- Range: [0.0, 1.0]

**Why it's valuable:**
- Model captures trend shape, but not breakout structure
- Donchian provides a clean, ML-friendly breakout signal
- Measures position within established trading range
- Identifies when price is near breakout levels

---

### 50. donchian_breakout
**What it represents:** Binary flag indicating breakout above prior 20-day high close (non-lookahead).

**How to Calculate (non-lookahead):**
1. `prior_20d_high_close = close.rolling(20, min_periods=20).max().shift(1)`
2. `donchian_breakout = (close > prior_20d_high_close).astype(int)`

**Normalization:** Binary (0 or 1)

**Feature Characteristics:**
- Uses prior 20-day highest CLOSE (not high), shifted by 1 bar to avoid lookahead bias
- Only uses information available before the current bar
- 0: Close is NOT above prior 20-day high close (no breakout)
- 1: Close IS above prior 20-day high close (breakout detected)
- Binary flag
- NaN values (from shift or insufficient data) are set to 0 (no breakout)

**Why it's valuable:**
- Model captures trend shape, but not breakout structure
- Donchian provides a clean, ML-friendly breakout signal
- Breakouts often precede strong trending moves
- Binary signal is easy for ML models to interpret
- Markets trend when breaking out of established ranges
- Non-lookahead implementation ensures no data leakage

---

## Trend Features (8)

### 51. trend_residual
**What it represents:** Deviation from linear trend (noise vs trend).

**How to Calculate:**
1. Fit linear regression to last 50 close values
2. Calculate residual: `(actual - fitted) / actual`
3. Take last residual value
4. Clip to `[-0.2, 0.2]`

**Normalization:** Clipped to [-0.2, 0.2]

**Feature Characteristics:**
- Negative: Price below trend (potential oversold)
- Positive: Price above trend (potential overbought)
- Near 0: Price follows trend closely
- Range: [-0.2, 0.2]

**Why it's valuable:** Measures how much price deviates from linear trend, useful for mean reversion.

---

### 52. adx14
**What it represents:** Average Directional Index (14-period) - trend strength indicator.

**How to Calculate:**
1. Calculate True Range: `TR = max(high-low, abs(high-prev_close), abs(low-prev_close))`
2. Calculate Directional Movement:
   - `+DM = today's high - yesterday high` (if positive and > -DM)
   - `-DM = yesterday low - today's low` (if positive and > +DM)
3. Smooth 14-day averages of +DM, -DM, and TR (Wilder's smoothing)
4. Calculate DI lines:
   - `+DI = 100 * (+DM14 / TR14)`
   - `-DI = 100 * (-DM14 / TR14)`
5. Calculate DX: `DX = 100 * abs(+DI - -DI) / (+DI + -DI)`
6. Calculate ADX: `ADX = EMA(DX, 14)`
7. Normalize: `adx14 = ADX / 100`
8. Clip to `[0, 1]`

**Normalization:** Normalized to [0, 1] range (ADX / 100)

**Feature Characteristics:**
- 0.0: No trend (ranging market)
- 0.25: Weak trend
- 0.50: Moderate trend
- 0.75: Strong trend
- 1.0: Very strong trend
- Range: [0.0, 1.0]
- Measures trend strength independent of direction
- One of the most predictive free indicators

**Why it's valuable:**
- Model currently knows trend direction (via slopes and HH/HL)
- But it does NOT know how strong the trend is
- ADX fills that gap perfectly
- Tells whether stock is trending strongly, ranging, losing momentum, or entering trend continuation

---

### 53. aroon_up
**What it represents:** Aroon Up (25-period) - normalized measure of days since highest high, indicating uptrend maturity.

**How to Calculate:**
1. Over a 25-period rolling window, find days since highest high
2. Calculate Aroon Up: `Aroon Up = 100 * (25 - days_since_highest_high) / 25`
3. Normalize: `aroon_up_norm = aroon_up / 100`
4. Clip to `[0, 1]`

**Normalization:** Normalized to [0, 1] range (Aroon Up / 100)

**Feature Characteristics:**
- 1.0: Highest high was today (fresh uptrend)
- 0.8: Highest high was 5 days ago (maturing uptrend)
- 0.4: Highest high was 15 days ago (aging uptrend)
- 0.0: Highest high was 25+ days ago (exhausted uptrend)
- Range: [0.0, 1.0]
- Measures uptrend freshness/maturity

**Why it's valuable:**
- Model currently knows if trend exists and how strong it is (ADX)
- But Aroon tells it how long the trend has been going on
- Trend age is often where swings succeed or fail
- Identifies if uptrend is fresh, maturing, or exhausted
- Fresh trends (high aroon_up) often continue, exhausted trends (low aroon_up) often reverse

---

### 54. aroon_down
**What it represents:** Aroon Down (25-period) - normalized measure of days since lowest low, indicating downtrend maturity.

**How to Calculate:**
1. Over a 25-period rolling window, find days since lowest low
2. Calculate Aroon Down: `Aroon Down = 100 * (25 - days_since_lowest_low) / 25`
3. Normalize: `aroon_down_norm = aroon_down / 100`
4. Clip to `[0, 1]`

**Normalization:** Normalized to [0, 1] range (Aroon Down / 100)

**Feature Characteristics:**
- 1.0: Lowest low was today (fresh downtrend)
- 0.8: Lowest low was 5 days ago (maturing downtrend)
- 0.4: Lowest low was 15 days ago (aging downtrend)
- 0.0: Lowest low was 25+ days ago (exhausted downtrend)
- Range: [0.0, 1.0]
- Measures downtrend freshness/maturity

**Why it's valuable:**
- Model currently knows if trend exists and how strong it is (ADX)
- But Aroon tells it how long the trend has been going on
- Trend age is often where swings succeed or fail
- Identifies if downtrend is starting or ending
- Fresh downtrends (high aroon_down) often continue, exhausted downtrends (low aroon_down) often reverse

---

### 55. aroon_oscillator
**What it represents:** Aroon Oscillator (25-period) - trend dominance indicator combining Aroon Up and Aroon Down.

**How to Calculate:**
1. Get normalized Aroon Up and Aroon Down (0-1 range)
2. Convert back to 0-100 range:
   - `aroon_up_raw = aroon_up * 100`
   - `aroon_down_raw = aroon_down * 100`
3. Calculate oscillator: `aroon_osc = aroon_up_raw - aroon_down_raw`
4. Normalize from [-100, 100] to [0, 1]:
   - `aroon_osc_norm = (aroon_osc + 100) / 200`
5. Clip to `[0, 1]`

**Normalization:** Normalized from [-100, 100] to [0, 1] range: `(aroon_osc + 100) / 200`

**Feature Characteristics:**
- 0.0: Strong downtrend dominance (aroon_osc = -100)
- 0.25: Moderate downtrend dominance (aroon_osc = -50)
- 0.5: Neutral/transition (aroon_osc = 0)
- 0.75: Moderate uptrend dominance (aroon_osc = +50)
- 1.0: Strong uptrend dominance (aroon_osc = +100)
- Range: [0.0, 1.0]
- Provides clean continuous signal for ML
- Net trend pressure measure

**Why it's valuable:**
- Captures trend dominance better than Up and Down alone
- Provides a clean continuous signal for ML
- Helps identify early trend reversals
- Combines both Aroon lines into single "net trend pressure" measure
- Positive values → uptrend dominance, negative values → downtrend dominance, near zero → trend transition

---

### 56. fractal_dimension_index
**What it represents:** Measures how "rough" the price path is (fractal dimension).

**How to Calculate:**
1. For each rolling window of N=100 prices:
   - Compute net displacement: `L_net = |P_N-1 - P_0|`
   - Compute path length: `L_path = sum(|P_i - P_i-1|)` for i=1 to N-1
   - Roughness ratio: `R = L_path / (L_net + ε)`
   - Fractal dimension: `FDI = 1 + log(R+1) / log(N)`
2. Normalize: Map from [1.0, 1.8] to [0, 1]: `FDI_norm = clip((FDI - 1.0) / (1.8 - 1.0), 0, 1)`

**Normalization:**
- FDI for financial time series typically lives in [1.0, 1.8]
- Normalized to [0, 1] using linear mapping

**Feature Characteristics:**
- FDI ≈ 0.0-0.4 (raw 1.0-1.3) → smooth, trending (trend-friendly environment)
- FDI ≈ 0.6 (raw 1.5) → borderline
- FDI ≈ 0.75-1.0 (raw 1.6-1.8) → choppy, mean-reverting, noisy (whipsaw environment)
- Range: [0, 1] (normalized)
- Tells model whether it's in a trend-friendly vs whipsaw environment

**Why it's valuable:**
- Directly encodes "is this tradable with trend-following or not"
- Helps model downweight momentum signals in very noisy regimes
- Pairs beautifully with ADX, Aroon, Donchian, TTM squeeze
- Very few retail systems use it – it's a genuine edge-type feature

---

### 57. hurst_exponent
**What it represents:** Quantifies whether returns persist, mean-revert, or act like noise (R/S method).

**How to Calculate:**
1. Compute log returns: `r_t = ln(P_t / P_{t-1})`
2. For each rolling window of N=100 returns:
   - Mean of returns: `μ`
   - Cumulative deviation series: `X_k = sum(r_i - μ)` for i=1 to k
   - Range: `R = max(X_k) - min(X_k)`
   - Standard deviation: `S = std(r_i)`
   - Rescaled range: `R/S`
   - Hurst estimate: `H ≈ log(R/S) / log(N)`
3. Clip to `[0, 1]`

**Normalization:**
- H naturally lives in [0, 1]
- Clipped to [0, 1] for ML use

**Feature Characteristics:**
- H > 0.5 → persistent/trending (moves tend to continue)
- H < 0.5 → mean-reverting (moves tend to snap back)
- H ≈ 0.5 → near-random walk
- Range: [0, 1]
- Tells model: should I expect continuation or snap-back after a move

**Why it's valuable:**
- Tells the model if momentum features should be trusted
- Great for swing trading where persistence matters
- Helps separate "fake breakouts" (H < 0.5, mean-reverting) from real trends
- Works great with existing ROC, RSI, Stoch, MACD/PPO, Donchian features

---

### 58. price_curvature
**What it represents:** Second derivative of trend (acceleration/deceleration).

**How to Calculate:**
1. Use SMA20 as smooth reference line: `T_t = SMA20(close)_t`
2. First derivative (slope): `S_t = T_t - T_{t-1}`
3. Second derivative (curvature): `C_t = S_t - S_{t-1}`
4. Normalize: `C_norm = clip(C_t / (close_t + ε), -0.05, 0.05)`

**Normalization:**
- Division by price makes it scale-invariant (comparable across tickers)
- Clipping to [-0.05, 0.05] limits insane spikes from gappy days

**Feature Characteristics:**
- Positive curvature → trend is bending up (acceleration)
- Negative curvature → trend is bending down (deceleration/topping)
- Near 0 → linear-ish trend, not bending much
- Range: [-0.05, 0.05] (normalized)

**Why it's valuable:**
- Distinguishes steady trends from accelerating/rolling-over ones
- Helps the model time entries inside an already-known trend
- Complements trend_residual, which measures deviation from a line, not curvature
- Very relevant for swing trading horizons

---

## Feature Summary by Category

| Category | Count | Features |
|----------|-------|----------|
| Price | 3 | price, price_log, price_vs_ma200 |
| Returns | 6 | daily_return, gap_pct, weekly_return_5d, monthly_return_21d, quarterly_return_63d, ytd_return |
| 52-Week | 3 | dist_52w_high, dist_52w_low, pos_52w |
| Moving Averages | 8 | sma20_ratio, sma50_ratio, sma200_ratio, sma20_sma50_ratio, sma50_sma200_ratio, sma50_slope, sma200_slope, kama_slope |
| Volatility | 8 | volatility_5d, volatility_21d, volatility_ratio, atr14_normalized, bollinger_band_width, ttm_squeeze_on, ttm_squeeze_momentum, volatility_of_volatility |
| Volume | 5 | log_volume, log_avg_volume_20d, relative_volume, chaikin_money_flow, obv_momentum |
| Momentum | 9 | rsi14, macd_histogram_normalized, ppo_histogram, dpo, roc10, roc20, stochastic_k14, cci20, williams_r14 |
| Market Context | 2 | beta_spy_252d, mkt_spy_dist_sma200 |
| Candlestick | 3 | candle_body_pct, candle_upper_wick_pct, candle_lower_wick_pct |
| Price Action | 5 | higher_high_10d, higher_low_10d, swing_low_10d, donchian_position, donchian_breakout |
| Trend | 8 | trend_residual, adx14, aroon_up, aroon_down, aroon_oscillator, fractal_dimension_index, hurst_exponent, price_curvature |
| **Total** | **60** | |

---

## Normalization Methods Summary

1. **Clipping**: Most features use clipping to cap extreme values (e.g., `clip(-0.2, 0.2)`)
2. **Log Transformation**: Volume features use `log1p()` to compress wide ranges
3. **Ratio Normalization**: Price ratios use division (e.g., `close / SMA200`)
4. **Centering**: RSI is centered around 50: `(rsi - 50) / 50`
5. **Division by Constant**: ADX divided by 100: `adx / 100`
6. **Tanh Compression**: CCI normalized with tanh: `tanh(cci / 100)`
7. **Binary**: Higher high/low and Donchian breakout features are binary (0 or 1)
8. **Already Normalized**: Some features are naturally in [0, 1] range (e.g., stochastic_k14, pos_52w, donchian_position)

---

## Best Practices

1. **Feature Selection**: All 60 features are enabled by default in `config/features.yaml`
2. **Training**: Use `config/train_features.yaml` to select features for model training
3. **Scaling**: Features are automatically scaled during training (StandardScaler for unbounded features)
4. **Validation**: Feature validation checks for infinities, excessive NaNs, and constant values
5. **Performance**: Feature computation is optimized with efficient DataFrame operations

---

## References

- Feature implementations: `features/technical.py`
- Feature registry: `features/registry.py`
- Feature configuration: `config/features.yaml`
- Training configuration: `config/train_features.yaml`

