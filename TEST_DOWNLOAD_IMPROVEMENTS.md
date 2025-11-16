# Testing Download Improvements

## Step 1: Install New Dependency

First, install the new `tqdm` package for progress bars:

```bash
pip install tqdm>=4.65.0
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Step 2: Quick Test with Small Dataset

Create a small test ticker file to test quickly:

```bash
# Create test ticker file
echo "AAPL" > data/tickers/test_tickers.csv
echo "MSFT" >> data/tickers/test_tickers.csv
echo "GOOGL" >> data/tickers/test_tickers.csv
```

Then test the download:
```bash
python src/download_data.py \
  --tickers-file data/tickers/test_tickers.csv \
  --start-date 2024-01-01 \
  --raw-folder data/raw \
  --sectors-file data/tickers/test_sectors.csv
```

**What to look for:**
- ✅ Progress bar appears
- ✅ Data validation messages (if any issues)
- ✅ Summary statistics at the end
- ✅ Files created in `data/raw/`

## Step 3: Test Resume Functionality

### Test 1: Interrupt and Resume
1. Start a download with many tickers:
```bash
python src/download_data.py \
  --tickers-file data/tickers/sp500_tickers.csv \
  --start-date 2024-01-01 \
  --raw-folder data/raw
```

2. **Interrupt it** (Ctrl+C) after a few tickers download

3. **Resume it:**
```bash
python src/download_data.py \
  --tickers-file data/tickers/sp500_tickers.csv \
  --start-date 2024-01-01 \
  --raw-folder data/raw \
  --resume
```

**What to look for:**
- ✅ Message: "Resuming: X tickers already completed"
- ✅ Skips already downloaded tickers
- ✅ Only downloads remaining tickers
- ✅ Checkpoint file exists: `data/raw/.download_checkpoint.json`

## Step 4: Test Retry Logic

Test retry by simulating network issues (harder to test, but you can verify it works):

```bash
# Test with max retries
python src/download_data.py \
  --tickers-file data/tickers/test_tickers.csv \
  --start-date 2024-01-01 \
  --max-retries 5 \
  --raw-folder data/raw
```

**What to look for:**
- ✅ If a download fails, you'll see retry messages
- ✅ "Attempt 1/4 failed: ... Retrying in 1.0s..."
- ✅ Eventually succeeds or reports final failure

## Step 5: Test Data Validation

The validation runs automatically. To see it in action:

1. Check existing data files for any issues
2. Re-download to trigger validation:
```bash
python src/download_data.py \
  --tickers-file data/tickers/test_tickers.csv \
  --start-date 2024-01-01 \
  --raw-folder data/raw \
  --full
```

**What to look for:**
- ✅ Warning messages if validation issues found
- ✅ "Data validation issues: ..." in logs
- ✅ Data still saved (with warnings)

## Step 6: Test Full Feature Set

Test all features together:

```bash
python src/swing_trade_app.py download \
  --tickers-file data/tickers/sp500_tickers.csv \
  --start-date 2024-01-01 \
  --max-retries 3 \
  --resume
```

**What to look for:**
- ✅ Progress bar with completion percentage
- ✅ Real-time statistics (Completed, Failed)
- ✅ Summary at end with all statistics
- ✅ Lists of failed tickers (if any)
- ✅ Data gaps reported (if any)

## Step 7: Verify Output

Check the downloaded files:

```bash
# Check files were created
dir data\raw\*.csv | measure-object | select-object Count

# Check a sample file
python -c "import pandas as pd; df = pd.read_csv('data/raw/AAPL.csv', index_col=0, parse_dates=True); print(f'AAPL: {len(df)} rows, Date range: {df.index.min()} to {df.index.max()}')"
```

**What to look for:**
- ✅ Files exist
- ✅ Data looks correct
- ✅ Date ranges are correct
- ✅ No obvious data quality issues

## Step 8: Test Summary Statistics

Run a download and check the summary:

```bash
python src/download_data.py \
  --tickers-file data/tickers/sp500_tickers.csv \
  --start-date 2024-01-01 \
  --raw-folder data/raw \
  --sectors-file data/tickers/sectors.csv
```

**What to look for in summary:**
- ✅ Total Tickers count
- ✅ Successfully Completed count
- ✅ Failed count (should be 0 or low)
- ✅ Up-to-date count (if re-running)
- ✅ Total Rows downloaded
- ✅ New Rows Added
- ✅ Validation Issues count
- ✅ Data Gaps Found count
- ✅ Lists of problematic tickers

## Step 9: Test Adaptive Rate Limiting

The adaptive rate limiting works automatically. To observe it:

1. Run a download with many tickers
2. Watch the pause times in logs
3. You should see: "Sleeping X.Xs before next chunk (adaptive)..."

**What to look for:**
- ✅ Pause times adjust based on chunk performance
- ✅ Faster chunks = shorter pauses
- ✅ Slower chunks = longer pauses

## Step 10: Test Error Handling

Test that errors are handled gracefully:

```bash
# Test with invalid ticker file (should handle gracefully)
python src/download_data.py \
  --tickers-file nonexistent_file.csv \
  --start-date 2024-01-01
```

**What to look for:**
- ✅ Clear error message
- ✅ Script exits gracefully
- ✅ No crashes or hangs

## Quick Test Script

Create a simple test script:

```bash
# test_download.bat
@echo off
echo Testing Download Improvements...
echo.

echo Step 1: Create test ticker file
echo AAPL > data\tickers\test_tickers.csv
echo MSFT >> data\tickers\test_tickers.csv
echo GOOGL >> data\tickers\test_tickers.csv

echo.
echo Step 2: Test download with all features
python src\download_data.py ^
  --tickers-file data\tickers\test_tickers.csv ^
  --start-date 2024-01-01 ^
  --raw-folder data\raw ^
  --sectors-file data\tickers\test_sectors.csv ^
  --max-retries 3

echo.
echo Step 3: Test resume (should skip already downloaded)
python src\download_data.py ^
  --tickers-file data\tickers\test_tickers.csv ^
  --start-date 2024-01-01 ^
  --raw-folder data\raw ^
  --sectors-file data\tickers\test_sectors.csv ^
  --resume

echo.
echo Test complete! Check data\raw\ for downloaded files.
pause
```

## Expected Results

### Successful Download Should Show:
1. **Progress Bar:**
   ```
   Downloading: 100%|████████████| 3/3 [00:15<00:00, Completed: 3, Failed: 0]
   ```

2. **Summary:**
   ```
   ================================================================================
   DOWNLOAD SUMMARY
   ================================================================================
   Total Tickers:        3
   Successfully Completed: 3
   Failed:              0
   Up-to-date (skipped): 0
   
   Data Statistics:
     Total Rows:        750
     New Rows Added:    750
     Validation Issues: 0
     Data Gaps Found:   0
   ================================================================================
   ```

3. **Files Created:**
   - `data/raw/AAPL.csv`
   - `data/raw/MSFT.csv`
   - `data/raw/GOOGL.csv`
   - `data/tickers/test_sectors.csv` (if sectors enabled)

## Troubleshooting

### If progress bar doesn't show:
- Check tqdm is installed: `pip install tqdm`
- Try running directly: `python src/download_data.py ...`

### If resume doesn't work:
- Check checkpoint file exists: `data/raw/.download_checkpoint.json`
- Make sure you're using `--resume` flag
- Check you're using the same `--raw-folder` path

### If validation issues appear:
- This is normal - some tickers may have data quality issues
- Check the specific issues reported
- Data is still saved (with warnings)

### If downloads fail:
- Check internet connection
- Try increasing `--max-retries`
- Check yfinance is working: `python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1d'))"`

## Next Steps After Testing

Once testing is successful:
1. Run full download on all tickers
2. Verify data quality
3. Proceed to next pipeline step (cleaning/features)

