"""
Service layer that wraps CLI functions for use in the GUI.
This provides a clean interface between the GUI and the existing CLI code.
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import joblib
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.identify_trades import (
    load_model as _load_model,
    identify_opportunities,
    get_recommended_filters,
    apply_entry_filters as _apply_entry_filters
)
from utils.stop_loss_policy import StopLossConfig


class TradeIdentificationService:
    """Service for identifying trading opportunities."""
    
    def __init__(self):
        self.model = None
        self.features = []
        self.scaler = None
        self.features_to_scale = []
        self.model_path = None
        
    def load_model(self, model_path: str) -> Tuple[bool, str]:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model pickle file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            path = Path(model_path)
            if not path.exists():
                return False, f"Model file not found: {model_path}"
            
            self.model, self.features, self.scaler, self.features_to_scale, _ = _load_model(path)
            self.model_path = str(path)
            
            msg = f"Model loaded successfully. Using {len(self.features)} features."
            if self.scaler is not None:
                msg += f" Scaler loaded for {len(self.features_to_scale)} features."
            
            return True, msg
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get list of available model files in the models directory."""
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.exists():
            return []
        
        model_files = list(models_dir.glob("*.pkl"))
        return [str(f.relative_to(PROJECT_ROOT)) for f in model_files]
    
    def identify_opportunities(
        self,
        data_dir: str = "data/features_labeled",
        tickers_file: str = "data/tickers/sp500_tickers.csv",
        min_probability: float = 0.5,
        top_n: int = 20,
        use_recommended_filters: bool = False,
        custom_filters: Optional[Dict[str, Tuple[str, float]]] = None,
        stop_loss_mode: Optional[str] = None,
        atr_stop_k: float = 1.8,
        atr_stop_min_pct: float = 0.04,
        atr_stop_max_pct: float = 0.10,
        swing_lookback_days: int = 10,
        swing_atr_buffer_k: float = 0.75
    ) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Identify trading opportunities.
        
        Returns:
            Tuple of (success: bool, opportunities: DataFrame or None, message: str)
        """
        if self.model is None:
            return False, None, "No model loaded. Please load a model first."
        
        try:
            # Load tickers
            tickers_df = pd.read_csv(tickers_file, header=None)
            tickers = tickers_df.iloc[:, 0].astype(str).tolist()
            
            # Build entry filters
            entry_filters = {}
            
            if use_recommended_filters:
                entry_filters.update(get_recommended_filters())
            
            if custom_filters:
                entry_filters.update(custom_filters)
            
            # Create stop-loss configuration
            stop_loss_config = None
            if stop_loss_mode in ["adaptive_atr", "swing_atr"]:
                stop_loss_config = StopLossConfig(
                    mode=stop_loss_mode,
                    atr_stop_k=atr_stop_k,
                    atr_stop_min_pct=atr_stop_min_pct,
                    atr_stop_max_pct=atr_stop_max_pct,
                    swing_lookback_days=swing_lookback_days,
                    swing_atr_buffer_k=swing_atr_buffer_k
                )
            
            # Identify opportunities
            opportunities = identify_opportunities(
                model=self.model,
                features=self.features,
                data_dir=Path(data_dir),
                tickers=tickers,
                min_probability=min_probability,
                top_n=top_n,
                scaler=self.scaler,
                features_to_scale=self.features_to_scale,
                entry_filters=entry_filters if entry_filters else None,
                stop_loss_config=stop_loss_config
            )
            
            if opportunities.empty:
                return True, opportunities, f"No opportunities found matching criteria (min_probability={min_probability:.1%})"
            
            return True, opportunities, f"Found {len(opportunities)} opportunities"
            
        except Exception as e:
            return False, None, f"Error identifying opportunities: {str(e)}"
    
    def get_recommended_filters(self) -> Dict[str, Tuple[str, float]]:
        """Get recommended entry filters."""
        return get_recommended_filters()


class DataService:
    """Service for data management operations."""
    
    @staticmethod
    def get_tickers_file() -> str:
        """Get default tickers file path."""
        return str(PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv")
    
    @staticmethod
    def get_data_dir() -> str:
        """Get default features data directory."""
        return str(PROJECT_ROOT / "data" / "features_labeled")
    
    @staticmethod
    def get_models_dir() -> str:
        """Get models directory."""
        return str(PROJECT_ROOT / "models")
    
    @staticmethod
    def get_raw_dir() -> str:
        """Get raw data directory."""
        return str(PROJECT_ROOT / "data" / "raw")
    
    @staticmethod
    def get_clean_dir() -> str:
        """Get cleaned data directory."""
        return str(PROJECT_ROOT / "data" / "clean")
    
    def get_total_tickers(self, tickers_file: str) -> int:
        """Get total number of tickers from file."""
        try:
            import pandas as pd
            df = pd.read_csv(tickers_file, header=None)
            return len(df)
        except Exception:
            return 0
    
    def get_download_progress(
        self, 
        raw_folder: str, 
        tickers_file: str, 
        full: bool,
        download_start_time: float = None,
        initial_count: int = 0
    ) -> Tuple[int, int]:
        """
        Get download progress by monitoring checkpoint file and counting files.
        
        Args:
            raw_folder: Directory with raw CSV files
            tickers_file: Path to tickers file
            full: Whether this is a full download (redownloads everything)
            download_start_time: Timestamp when download started (for full downloads)
            initial_count: Number of files that existed before download started
        
        Returns:
            Tuple of (completed: int, total: int)
        """
        import json
        import os
        
        total = self.get_total_tickers(tickers_file)
        if total == 0:
            return 0, 0
        
        raw_path = Path(raw_folder)
        checkpoint_file = raw_path / '.download_checkpoint.json'
        
        # Count existing CSV files
        completed_count = 0
        if raw_path.exists():
            try:
                if full:
                    # For full downloads, only count files modified AFTER download started
                    # This ensures we start at 0 and only count newly downloaded files
                    if download_start_time is not None:
                        csv_files = list(raw_path.glob("*.csv"))
                        for csv_file in csv_files:
                            try:
                                mtime = os.path.getmtime(csv_file)
                                # Only count files modified after download started
                                # Add small buffer (1 second) to account for timing issues
                                if mtime >= (download_start_time - 1.0):
                                    completed_count += 1
                            except Exception:
                                # If we can't get mtime, don't count it for full downloads
                                # (safer to undercount than overcount)
                                pass
                    else:
                        # If we don't have start time, can't track properly - return 0
                        completed_count = 0
                else:
                    # For normal/resume downloads, count all files
                    completed_count = len(list(raw_path.glob("*.csv")))
            except Exception:
                completed_count = 0
        
        # If checkpoint exists, use it (might be more accurate for resume)
        if checkpoint_file.exists() and not full:
            try:
                with open(checkpoint_file, 'r') as f:
                    completed = json.load(f)
                    if isinstance(completed, list):
                        checkpoint_count = len(completed)
                        # Use the higher of the two (checkpoint might lag behind file count)
                        completed_count = max(completed_count, checkpoint_count)
            except Exception:
                pass
        
        return completed_count, total
    
    def download_data(
        self,
        tickers_file: str = None,
        start_date: str = "2008-01-01",
        raw_folder: str = None,
        sectors_file: str = None,
        full: bool = False,
        resume: bool = False,
        max_retries: int = 3,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Download stock data.
        
        Args:
            progress_callback: Optional function(completed, total) to call for progress updates
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        import subprocess
        import sys
        import time
        import threading
        
        if tickers_file is None:
            tickers_file = self.get_tickers_file()
        if raw_folder is None:
            raw_folder = self.get_raw_dir()
        if sectors_file is None:
            sectors_file = str(PROJECT_ROOT / "data" / "tickers" / "sectors.csv")
        
        total = self.get_total_tickers(tickers_file)
        
        # Get initial file count (before download starts)
        initial_count = 0
        raw_path = Path(raw_folder)
        if raw_path.exists():
            initial_count = len(list(raw_path.glob("*.csv")))
        
        # For full downloads, we need to track files that existed before vs new ones
        # We'll use file modification time - files modified after download starts are "new"
        download_start_time = time.time()
        
        scripts_dir = PROJECT_ROOT / "src"
        cmd = [
            sys.executable,
            str(scripts_dir / "download_data.py"),
            "--tickers-file", tickers_file,
            "--start-date", start_date,
            "--raw-folder", raw_folder,
            "--sectors-file", sectors_file,
            "--chunk-size", "100",
            "--pause", "1.0",
            "--max-retries", str(max_retries)
        ]
        
        if full:
            cmd.append("--full")
        if resume:
            cmd.append("--resume")
        
        try:
            # Start subprocess - capture output to parse failure information
            # We'll read it in background threads to prevent blocking
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,  # Capture stdout to parse summary
                stderr=subprocess.PIPE,  # Keep stderr for error messages
                text=True,
                bufsize=1  # Line buffered
            )
            
            stdout_lines = []
            stderr_lines = []
            
            def read_stdout():
                """Read stdout in background to prevent blocking."""
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        stdout_lines.append(line)
                        # Flush to ensure we get all output
                        if not process.poll() is None:
                            # Process ended, read any remaining data
                            remaining = process.stdout.read()
                            if remaining:
                                stdout_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            def read_stderr():
                """Read stderr in background to prevent blocking."""
                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        stderr_lines.append(line)
                        # Flush to ensure we get all output
                        if not process.poll() is None:
                            # Process ended, read any remaining data
                            remaining = process.stderr.read()
                            if remaining:
                                stderr_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor progress
            # For full downloads, we start from 0 (will redownload everything)
            # For resume/normal, we account for existing files
            if full:
                base_count = 0  # Full download starts fresh
            else:
                base_count = initial_count  # Normal/resume accounts for existing
            
            last_count = base_count
            no_progress_count = 0
            max_no_progress = 60  # Allow up to 30 seconds (60 * 0.5s) without progress
            
            if progress_callback:
                while process.poll() is None:
                    # Get current progress
                    current_count, total_count = self.get_download_progress(
                        raw_folder, 
                        tickers_file, 
                        full,
                        download_start_time=download_start_time if full else None,
                        initial_count=initial_count
                    )
                    
                    # For full downloads, current_count already only counts new files
                    # For normal/resume, adjust for base count
                    if full:
                        adjusted_count = current_count  # Already only new files
                    else:
                        adjusted_count = max(0, current_count - base_count)
                    
                    if total_count > 0:
                        progress_callback(adjusted_count, total_count)
                    
                    # Check if progress is stuck
                    if current_count == last_count:
                        no_progress_count += 1
                    else:
                        no_progress_count = 0
                        last_count = current_count
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
            
            # Wait for completion
            process.wait()
            
            # Wait for threads to finish reading (give them more time)
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            
            # Small delay to ensure all buffered output is captured
            time.sleep(0.2)
            
            # Get final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            # If stdout is empty, the output might have gone to stderr (common with logging)
            # Try to parse from stderr as well
            if not stdout and stderr:
                # Sometimes the summary is in stderr if logging is configured that way
                pass  # We'll parse both anyway
            
            if process.returncode == 0:
                # Final progress update
                if progress_callback:
                    final_count, total_count = self.get_download_progress(
                        raw_folder, 
                        tickers_file, 
                        full,
                        download_start_time=download_start_time if full else None,
                        initial_count=initial_count
                    )
                    if full:
                        adjusted_count = final_count  # Already only new files
                    else:
                        adjusted_count = max(0, final_count - base_count)
                    progress_callback(adjusted_count, total_count)
                
                # Parse output to check for failures
                message = self._parse_download_summary(stdout, stderr)
                return True, message
            else:
                error_msg = stderr if stderr else stdout if stdout else f"Process exited with code {process.returncode}"
                return False, f"Download failed: {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _parse_download_summary(self, stdout: str, stderr: str) -> str:
        """
        Parse download output to extract statistics.
        
        The download script prints a summary like:
        ================================================================================
        DOWNLOAD SUMMARY
        ================================================================================
        Total Tickers:        500
        Successfully Completed: 495
        Failed:              5
        Up-to-date (skipped): 10
        ...
        Failed Tickers (5):
          - TICKER1
          - TICKER2
          ...
        
        Returns:
            Success message with statistics: (x Completed, x Skipped, x Failed)
        """
        import re
        
        # Combine stdout and stderr
        output = stdout + stderr
        
        # Extract statistics from the summary
        total_completed = 0  # Total successfully processed (includes skipped)
        skipped_count = 0
        failed_count = 0
        
        # Try to find "Successfully Completed: X" pattern
        # Note: This includes both newly downloaded AND up-to-date (skipped) files
        completed_match = re.search(r'Successfully Completed:\s*(\d+)', output, re.IGNORECASE)
        if completed_match:
            total_completed = int(completed_match.group(1))
        
        # Try to find "Up-to-date (skipped): X" pattern
        skipped_match = re.search(r'Up-to-date\s*\(skipped\):\s*(\d+)', output, re.IGNORECASE)
        if skipped_match:
            skipped_count = int(skipped_match.group(1))
        
        # Try to find "Failed: X" pattern
        failed_match = re.search(r'Failed:\s*(\d+)', output, re.IGNORECASE)
        if failed_match:
            failed_count = int(failed_match.group(1))
        
        # Calculate actual new downloads (completed minus skipped)
        # The "Successfully Completed" count includes both new downloads and skipped files
        actual_completed = max(0, total_completed - skipped_count) if total_completed > 0 else 0
        
        # Build message with statistics
        stats_parts = []
        if actual_completed > 0:
            stats_parts.append(f"{actual_completed} Completed")
        if skipped_count > 0:
            stats_parts.append(f"{skipped_count} Skipped")
        if failed_count > 0:
            stats_parts.append(f"{failed_count} Failed")
        
        if stats_parts:
            message = f"Download completed successfully ({', '.join(stats_parts)})"
        else:
            message = "Download completed successfully"
        
        return message
    
    def get_total_raw_files(self, raw_dir: str) -> int:
        """Get total number of CSV files in raw directory."""
        try:
            raw_path = Path(raw_dir)
            if raw_path.exists():
                return len(list(raw_path.glob("*.csv")))
            return 0
        except Exception:
            return 0
    
    def get_clean_progress(
        self, 
        raw_dir: str, 
        clean_dir: str, 
        full: bool,
        clean_start_time: float = None
    ) -> Tuple[int, int]:
        """
        Get cleaning progress by counting cleaned files.
        
        Args:
            raw_dir: Directory with raw CSV files
            clean_dir: Directory with cleaned Parquet files
            full: Whether this is a full clean (reclean all files)
            clean_start_time: Timestamp when cleaning started (for full clean)
        
        Returns:
            Tuple of (cleaned: int, total: int)
        """
        import os
        
        total = self.get_total_raw_files(raw_dir)
        if total == 0:
            return 0, 0
        
        clean_path = Path(clean_dir)
        cleaned_count = 0
        if clean_path.exists():
            try:
                if full and clean_start_time is not None:
                    # For full clean, only count files modified after clean started
                    parquet_files = list(clean_path.glob("*.parquet"))
                    for parquet_file in parquet_files:
                        try:
                            mtime = os.path.getmtime(parquet_file)
                            # Only count files modified after clean started
                            # Add small buffer (1 second) to account for timing issues
                            if mtime >= (clean_start_time - 1.0):
                                cleaned_count += 1
                        except Exception:
                            # If we can't get mtime, don't count it for full clean
                            pass
                else:
                    # For normal/resume clean, count all files
                    cleaned_count = len(list(clean_path.glob("*.parquet")))
            except Exception:
                cleaned_count = 0
        
        return cleaned_count, total
    
    def clean_data(
        self,
        raw_dir: str = None,
        clean_dir: str = None,
        full: bool = False,
        resume: bool = False,
        workers: int = 4,
        verbose: bool = False,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Clean raw CSV files into Parquet format.
        
        Args:
            progress_callback: Optional function(completed, total) to call for progress updates
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        import subprocess
        import sys
        import time
        import threading
        
        if raw_dir is None:
            raw_dir = self.get_raw_dir()
        if clean_dir is None:
            clean_dir = self.get_clean_dir()
        
        # Get initial count of cleaned files (before cleaning starts)
        initial_cleaned = 0
        clean_path = Path(clean_dir)
        if clean_path.exists() and not full:
            # For full clean, we start from 0 (will reclean everything)
            initial_cleaned = len(list(clean_path.glob("*.parquet")))
        
        total_files = self.get_total_raw_files(raw_dir)
        
        # For full clean, we need to track files that existed before vs new ones
        # We'll use file modification time - files modified after clean starts are "new"
        clean_start_time = time.time()
        
        scripts_dir = PROJECT_ROOT / "src"
        cmd = [
            sys.executable,
            str(scripts_dir / "clean_data.py"),
            "--raw-dir", raw_dir,
            "--clean-dir", clean_dir,
            "--workers", str(workers)
        ]
        
        # For full clean, we don't use --resume (will reclean everything)
        # We'll handle it by deleting existing files or letting the script overwrite them
        if full:
            # Delete existing cleaned files to force re-cleaning
            clean_path = Path(clean_dir)
            if clean_path.exists():
                try:
                    for parquet_file in clean_path.glob("*.parquet"):
                        try:
                            parquet_file.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
        elif resume:
            cmd.append("--resume")
        
        if verbose:
            cmd.append("--verbose")
        
        try:
            # Start subprocess - capture output to parse summary
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,  # Capture stdout to parse summary
                stderr=subprocess.PIPE,  # Keep stderr for error messages
                text=True,
                bufsize=1  # Line buffered
            )
            
            stdout_lines = []
            stderr_lines = []
            
            def read_stdout():
                """Read stdout in background to prevent blocking."""
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        stdout_lines.append(line)
                        if not process.poll() is None:
                            remaining = process.stdout.read()
                            if remaining:
                                stdout_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            def read_stderr():
                """Read stderr in background to prevent blocking."""
                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        stderr_lines.append(line)
                        if not process.poll() is None:
                            remaining = process.stderr.read()
                            if remaining:
                                stderr_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor progress
            if progress_callback:
                while process.poll() is None:
                    # Get current progress
                    current_cleaned, total_count = self.get_clean_progress(
                        raw_dir, 
                        clean_dir, 
                        full,
                        clean_start_time=clean_start_time if full else None
                    )
                    
                    # Adjust for initial count
                    if full:
                        # For full clean, current_cleaned already only counts new files
                        adjusted_count = current_cleaned
                    elif resume:
                        # For resume, we only count new files cleaned
                        adjusted_count = max(0, current_cleaned - initial_cleaned)
                    else:
                        # For fresh clean, count all cleaned files
                        adjusted_count = current_cleaned
                    
                    if total_count > 0:
                        progress_callback(adjusted_count, total_count)
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
            
            # Wait for completion
            process.wait()
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            
            # Small delay to ensure all buffered output is captured
            time.sleep(0.2)
            
            # Get final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            if process.returncode == 0:
                # Final progress update
                if progress_callback:
                    final_cleaned, total_count = self.get_clean_progress(
                        raw_dir, 
                        clean_dir, 
                        full,
                        clean_start_time=clean_start_time if full else None
                    )
                    if full:
                        adjusted_count = final_cleaned  # Already only new files
                    elif resume:
                        adjusted_count = max(0, final_cleaned - initial_cleaned)
                    else:
                        adjusted_count = final_cleaned
                    progress_callback(adjusted_count, total_count)
                
                # Parse output to get statistics
                message = self._parse_clean_summary(stdout, stderr)
                return True, message
            else:
                error_msg = stderr if stderr else stdout if stdout else f"Process exited with code {process.returncode}"
                return False, f"Cleaning failed: {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _parse_clean_summary(self, stdout: str, stderr: str) -> str:
        """
        Parse clean data output to extract statistics.
        
        Returns:
            Success message with statistics
        """
        import re
        
        # Combine stdout and stderr
        output = stdout + stderr
        
        # Look for summary statistics
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Try to find patterns in the output
        # The clean script may print summary statistics
        success_match = re.search(r'(?:Successfully|Processed|Cleaned):\s*(\d+)', output, re.IGNORECASE)
        if success_match:
            success_count = int(success_match.group(1))
        
        failed_match = re.search(r'(?:Failed|Errors):\s*(\d+)', output, re.IGNORECASE)
        if failed_match:
            failed_count = int(failed_match.group(1))
        
        skipped_match = re.search(r'(?:Skipped|Already cleaned):\s*(\d+)', output, re.IGNORECASE)
        if skipped_match:
            skipped_count = int(skipped_match.group(1))
        
        # Build message with statistics
        stats_parts = []
        if success_count > 0:
            stats_parts.append(f"{success_count} Cleaned")
        if skipped_count > 0:
            stats_parts.append(f"{skipped_count} Skipped")
        if failed_count > 0:
            stats_parts.append(f"{failed_count} Failed")
        
        if stats_parts:
            message = f"Data cleaning completed successfully ({', '.join(stats_parts)})"
        else:
            message = "Data cleaning completed successfully"
        
        return message


class FeatureService:
    """Service for feature engineering operations."""
    
    def get_output_dir(self) -> str:
        """Get default output directory for features."""
        return str(PROJECT_ROOT / "data" / "features_labeled")
    
    def get_total_input_files(self, input_dir: str) -> int:
        """Get total number of Parquet files in input directory."""
        try:
            input_path = Path(input_dir)
            if input_path.exists():
                return len(list(input_path.glob("*.parquet")))
            return 0
        except Exception:
            return 0
    
    def get_feature_progress(
        self, 
        input_dir: str, 
        output_dir: str, 
        full: bool,
        feature_start_time: float = None
    ) -> Tuple[int, int]:
        """
        Get feature building progress by counting output files.
        
        Args:
            input_dir: Directory with input Parquet files
            output_dir: Directory with output feature Parquet files
            full: Whether this is a full rebuild
            feature_start_time: Timestamp when feature building started (for full rebuild)
        
        Returns:
            Tuple of (completed: int, total: int)
        """
        import os
        
        total = self.get_total_input_files(input_dir)
        if total == 0:
            return 0, 0
        
        output_path = Path(output_dir)
        completed_count = 0
        if output_path.exists():
            try:
                if full and feature_start_time is not None:
                    # For full rebuild, only count files modified after build started
                    parquet_files = list(output_path.glob("*.parquet"))
                    for parquet_file in parquet_files:
                        try:
                            mtime = os.path.getmtime(parquet_file)
                            # Only count files modified after build started
                            if mtime >= (feature_start_time - 1.0):
                                completed_count += 1
                        except Exception:
                            pass
                else:
                    # For normal build, count all files
                    completed_count = len(list(output_path.glob("*.parquet")))
            except Exception:
                completed_count = 0
        
        return completed_count, total
    
    def build_features(
        self,
        input_dir: str = None,
        output_dir: str = None,
        config: str = None,
        horizon: int = 5,
        threshold: float = 0.0,
        full: bool = False,
        feature_set: Optional[str] = None,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Build feature set from cleaned data.
        
        Args:
            progress_callback: Optional function(completed, total) to call for progress updates
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        import subprocess
        import sys
        import time
        import threading
        
        if input_dir is None:
            input_dir = str(PROJECT_ROOT / "data" / "clean")
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "data" / "features_labeled")
        if config is None:
            config = str(PROJECT_ROOT / "config" / "features.yaml")
        
        # Get initial count of feature files (before building starts)
        initial_features = 0
        output_path = Path(output_dir)
        if output_path.exists() and not full:
            initial_features = len(list(output_path.glob("*.parquet")))
        
        total_files = self.get_total_input_files(input_dir)
        
        # For full rebuild, we need to track files that existed before vs new ones
        feature_start_time = time.time()
        
        scripts_dir = PROJECT_ROOT / "src"
        cmd = [
            sys.executable,
            str(scripts_dir / "feature_pipeline.py"),
            "--horizon", str(horizon),
            "--threshold", str(threshold)
        ]
        
        if feature_set:
            cmd.extend(["--feature-set", feature_set])
            if input_dir != str(PROJECT_ROOT / "data" / "clean"):
                cmd.extend(["--input-dir", input_dir])
        else:
            cmd.extend([
                "--input-dir", input_dir,
                "--output-dir", output_dir,
                "--config", config
            ])
        
        if full:
            cmd.append("--full")
        
        try:
            # Start subprocess - capture output to parse summary
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,  # Capture stdout to parse summary
                stderr=subprocess.PIPE,  # Keep stderr for error messages
                text=True,
                bufsize=1  # Line buffered
            )
            
            stdout_lines = []
            stderr_lines = []
            
            def read_stdout():
                """Read stdout in background to prevent blocking."""
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        stdout_lines.append(line)
                        if not process.poll() is None:
                            remaining = process.stdout.read()
                            if remaining:
                                stdout_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            def read_stderr():
                """Read stderr in background to prevent blocking."""
                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        stderr_lines.append(line)
                        if not process.poll() is None:
                            remaining = process.stderr.read()
                            if remaining:
                                stderr_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor progress
            if progress_callback:
                while process.poll() is None:
                    # Get current progress
                    current_features, total_count = self.get_feature_progress(
                        input_dir, 
                        output_dir, 
                        full,
                        feature_start_time=feature_start_time if full else None
                    )
                    
                    # Adjust for initial count
                    if full:
                        # For full rebuild, current_features already only counts new files
                        adjusted_count = current_features
                    else:
                        # For normal build, only count new files
                        adjusted_count = max(0, current_features - initial_features)
                    
                    if total_count > 0:
                        progress_callback(adjusted_count, total_count)
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
            
            # Wait for completion
            process.wait()
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            
            # Small delay to ensure all buffered output is captured
            time.sleep(0.2)
            
            # Get final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            if process.returncode == 0:
                # Final progress update
                if progress_callback:
                    final_features, total_count = self.get_feature_progress(
                        input_dir, 
                        output_dir, 
                        full,
                        feature_start_time=feature_start_time if full else None
                    )
                    if full:
                        adjusted_count = final_features  # Already only new files
                    else:
                        adjusted_count = max(0, final_features - initial_features)
                    progress_callback(adjusted_count, total_count)
                
                # Parse output to get statistics
                message = self._parse_feature_summary(stdout, stderr, horizon, threshold)
                return True, message
            else:
                error_msg = stderr if stderr else stdout if stdout else f"Process exited with code {process.returncode}"
                return False, f"Feature building failed: {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _parse_feature_summary(self, stdout: str, stderr: str, horizon: int, threshold: float) -> str:
        """
        Parse feature building output to extract statistics.
        
        Returns:
            Success message with statistics
        """
        import re
        
        # Combine stdout and stderr
        output = stdout + stderr
        
        # Look for summary statistics
        total_count = 0
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        # Try to find patterns in the output
        total_match = re.search(r'Total tickers:\s*(\d+)', output, re.IGNORECASE)
        if total_match:
            total_count = int(total_match.group(1))
        
        processed_match = re.search(r'Processed:\s*(\d+)', output, re.IGNORECASE)
        if processed_match:
            processed_count = int(processed_match.group(1))
        
        skipped_match = re.search(r'Skipped.*?:\s*(\d+)', output, re.IGNORECASE)
        if skipped_match:
            skipped_count = int(skipped_match.group(1))
        
        failed_match = re.search(r'Failed:\s*(\d+)', output, re.IGNORECASE)
        if failed_match:
            failed_count = int(failed_match.group(1))
        
        # Build message with statistics
        stats_parts = []
        if processed_count > 0:
            stats_parts.append(f"{processed_count} Processed")
        if skipped_count > 0:
            stats_parts.append(f"{skipped_count} Skipped")
        if failed_count > 0:
            stats_parts.append(f"{failed_count} Failed")
        
        if stats_parts:
            message = f"Features built successfully ({', '.join(stats_parts)})"
        else:
            message = f"Features built successfully (horizon={horizon}d, threshold={threshold:.2%})"
        
        return message


class TrainingService:
    """Service for model training operations."""
    
    def train_model(
        self,
        tune: bool = False,
        n_iter: int = 20,
        cv: bool = False,
        no_early_stop: bool = False,
        plots: bool = False,
        fast: bool = False,
        cv_folds: Optional[int] = None,
        diagnostics: bool = False,
        imbalance_multiplier: float = 1.0,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        val_end: Optional[str] = None,
        horizon: Optional[int] = None,
        label_col: Optional[str] = None,
        feature_set: Optional[str] = None,
        model_output: Optional[str] = None,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Train ML model.
        
        Args:
            progress_callback: Optional function(stage, total_stages, message) to call for progress updates
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        import subprocess
        import sys
        import time
        import threading
        import re
        
        scripts_dir = PROJECT_ROOT / "src"
        train_script = scripts_dir / "train_model.py"
        
        # Verify script exists
        if not train_script.exists():
            return False, f"Training script not found: {train_script}"
        
        cmd = [sys.executable, str(train_script)]
        
        if tune:
            cmd.append("--tune")
            cmd.extend(["--n-iter", str(n_iter)])
        if cv:
            cmd.append("--cv")
        if no_early_stop:
            cmd.append("--no-early-stop")
        if plots:
            cmd.append("--plots")
        if fast:
            cmd.append("--fast")
        if cv_folds is not None:
            cmd.extend(["--cv-folds", str(cv_folds)])
        if diagnostics:
            cmd.append("--diagnostics")
        if imbalance_multiplier != 1.0:
            cmd.extend(["--imbalance-multiplier", str(imbalance_multiplier)])
        if train_end is not None:
            cmd.extend(["--train-end", train_end])
        if val_end is not None:
            cmd.extend(["--val-end", val_end])
        if horizon is not None:
            cmd.extend(["--horizon", str(horizon)])
        if label_col is not None:
            cmd.extend(["--label-col", label_col])
        if feature_set is not None:
            cmd.extend(["--feature-set", feature_set])
        if model_output is not None:
            cmd.extend(["--model-output", model_output])
        
        # Define training stages (in order)
        # Patterns are matched case-insensitively against output lines
        stages = [
            ("Loading data", "Loading data|load_data|Loading feature"),
            ("Preparing data", "Preparing data|SANITY CHECKS|X shape|Label distribution|prepare\\(|enabled_feats"),
            ("Training baseline", "TRAINING BASELINE|Baseline|DummyClassifier|LogisticRegression"),
            ("Hyperparameter tuning", "hyperparameter tuning|Performing hyperparameter|RandomizedSearchCV|Fitting|n_iter"),
            ("Training XGBoost", "TRAINING XGBOOST|Retraining|Training with early stopping|XGBClassifier|best_iteration"),
            ("Evaluating model", "Evaluating|Evaluation|ROC AUC|Test set|Validation set|precision|recall"),
            ("Saving model", "Saving|Model saved|TRAINING COMPLETE|metadata|joblib.dump"),
            ("SHAP diagnostics", "SHAP DIAGNOSTICS|SHAP|shap values|shap_importances")
        ]
        
        # Adjust total stages based on options
        total_stages = 7  # Base stages
        if tune:
            total_stages += 1  # Hyperparameter tuning is a separate stage
        if diagnostics:
            total_stages += 1  # SHAP diagnostics is a separate stage
        
        try:
            # Use simpler approach: run with real-time output capture
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
            
            # Start subprocess - merge stderr into stdout for easier reading
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                env=env,
                cwd=str(PROJECT_ROOT),
                universal_newlines=True
            )
            
            stdout_lines = []
            stderr_lines = []
            current_stage = 0
            
            def read_stdout():
                """Read stdout in background to prevent blocking."""
                nonlocal current_stage
                try:
                    import select
                    import sys
                    
                    # For Windows, use iter() with readline which handles blocking better
                    if sys.platform == 'win32':
                        # Windows: use iter() with readline for non-blocking behavior
                        try:
                            for line in iter(process.stdout.readline, ''):
                                if line:
                                    stdout_lines.append(line)
                                    # Check for stage progression
                                    if progress_callback:
                                        for i, (stage_name, patterns) in enumerate(stages):
                                            if i >= current_stage:
                                                pattern_list = patterns.split("|")
                                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in pattern_list):
                                                    if i > current_stage:
                                                        current_stage = i
                                                        progress_callback(current_stage + 1, total_stages, stage_name)
                                                    break
                        except Exception:
                            # Fallback: read all at once after process completes
                            pass
                    else:
                        # Unix: use select for non-blocking reads
                        while process.poll() is None:
                            ready, _, _ = select.select([process.stdout], [], [], 0.1)
                            if ready:
                                line = process.stdout.readline()
                                if line:
                                    stdout_lines.append(line)
                                    # Check for stage progression
                                    if progress_callback:
                                        for i, (stage_name, patterns) in enumerate(stages):
                                            if i >= current_stage:
                                                pattern_list = patterns.split("|")
                                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in pattern_list):
                                                    if i > current_stage:
                                                        current_stage = i
                                                        progress_callback(current_stage + 1, total_stages, stage_name)
                                                    break
                    
                    # Read any remaining data after process ends
                    try:
                        remaining = process.stdout.read()
                        if remaining:
                            stdout_lines.append(remaining)
                    except Exception:
                        pass
                except Exception as e:
                    # Log error but don't break the process
                    import sys
                    print(f"Error reading stdout: {e}", file=sys.stderr)
            
            def read_stderr():
                """Read stderr in background to prevent blocking."""
                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        stderr_lines.append(line)
                        if not process.poll() is None:
                            remaining = process.stderr.read()
                            if remaining:
                                stderr_lines.append(remaining)
                            break
                except Exception:
                    pass
            
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Initial progress
            if progress_callback:
                progress_callback(1, total_stages, "Starting training...")
            
            # Give process a moment to start
            time.sleep(0.5)
            
            # Check if process started successfully (didn't exit immediately)
            if process.poll() is not None:
                # Process ended immediately - likely an error
                stdout_thread.join(timeout=1.0)
                stdout = ''.join(stdout_lines)
                stderr = ''.join(stderr_lines)
                error_msg = (stdout + stderr).strip() if (stdout or stderr) else "Process exited immediately with no output"
                return False, f"Training failed to start: {error_msg}"
            
            # Wait for process to complete
            # Use a simple wait with periodic checks for progress updates
            start_time = time.time()
            last_progress_update = start_time
            
            while process.poll() is None:
                # Update progress periodically even if we're not detecting stages
                elapsed = time.time() - start_time
                if elapsed > 3600:  # 1 hour timeout
                    process.kill()
                    return False, "Training timed out after 1 hour"
                
                # If we have output but haven't updated progress in a while, show we're working
                if len(stdout_lines) > 0 and time.time() - last_progress_update > 10:
                    if progress_callback:
                        # Show we're making progress even if stage detection isn't working
                        progress_callback(
                            min(current_stage + 2, total_stages), 
                            total_stages, 
                            f"Training in progress... ({len(stdout_lines)} lines processed)"
                        )
                        last_progress_update = time.time()
                
                time.sleep(0.5)  # Check every 0.5 seconds
            
            # Process finished
            returncode = process.returncode
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=3.0)
            stderr_thread.join(timeout=3.0)
            
            # Small delay to ensure all buffered output is captured
            time.sleep(0.3)
            
            # Get final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            if returncode == 0:
                # Final progress update
                if progress_callback:
                    progress_callback(total_stages, total_stages, "Training completed")
                
                # Parse output to get model info
                message = self._parse_training_summary(stdout, stderr)
                return True, message
            else:
                # Include both stdout and stderr in error message for debugging
                error_parts = []
                if stderr:
                    error_parts.append(f"STDERR: {stderr[-2000:]}")  # Last 2000 chars
                if stdout:
                    error_parts.append(f"STDOUT: {stdout[-2000:]}")  # Last 2000 chars
                if not error_parts:
                    error_parts.append(f"Process exited with code {returncode}")
                
                error_msg = " | ".join(error_parts)
                return False, f"Training failed (exit code {returncode}): {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _parse_training_summary(self, stdout: str, stderr: str) -> str:
        """
        Parse training output to extract summary information.
        
        Returns:
            Success message with training details
        """
        import re
        
        # Combine stdout and stderr
        output = stdout + stderr
        
        # Look for model file path
        model_path_match = re.search(r'Model saved to:\s*(.+)', output, re.IGNORECASE)
        model_path = model_path_match.group(1).strip() if model_path_match else None
        
        # Look for training time
        time_match = re.search(r'Total training time:\s*([\d.]+)\s*seconds', output, re.IGNORECASE)
        training_time = time_match.group(1) if time_match else None
        
        # Build message
        message = "Model training completed successfully"
        if model_path:
            message += f" - Model saved to: {Path(model_path).name}"
        if training_time:
            minutes = float(training_time) / 60
            message += f" ({minutes:.1f} minutes)"
        
        return message


class BacktestService:
    """Service for backtesting operations."""
    
    def run_backtest(
        self,
        horizon: int,
        return_threshold: float,
        position_size: float = 1000.0,
        strategy: str = "model",
        model_path: str = None,
        model_threshold: float = 0.5,
        stop_loss: Optional[float] = None,
        stop_loss_mode: Optional[str] = None,
        atr_stop_k: float = 1.8,
        atr_stop_min_pct: float = 0.04,
        atr_stop_max_pct: float = 0.10,
        output: Optional[str] = None,
        entry_filters: Optional[list] = None,
        progress_callback=None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Run backtest.
        
        Args:
            progress_callback: Optional function(completed, total, message) to call for progress updates
        
        Returns:
            Tuple of (success: bool, output_file: str or None, message: str)
        """
        import subprocess
        import sys
        import time
        import threading
        import re
        
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "xgb_classifier_selected_features.pkl")
        
        scripts_dir = PROJECT_ROOT / "src"
        cmd = [
            sys.executable,
            str(scripts_dir / "enhanced_backtest.py"),
            "--horizon", str(horizon),
            "--return-threshold", str(return_threshold),
            "--position-size", str(position_size),
            "--strategy", strategy,
            "--model", model_path,
            "--model-threshold", str(model_threshold)
        ]
        
        if stop_loss is not None:
            cmd.extend(["--stop-loss", str(stop_loss)])
        if stop_loss_mode is not None:
            cmd.extend(["--stop-loss-mode", stop_loss_mode])
            cmd.extend(["--atr-stop-k", str(atr_stop_k)])
            cmd.extend(["--atr-stop-min-pct", str(atr_stop_min_pct)])
            cmd.extend(["--atr-stop-max-pct", str(atr_stop_max_pct)])
        
        output_path = output
        if output_path:
            cmd.extend(["--output", output_path])
        
        try:
            # Start subprocess - capture output to parse progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            stdout_lines = []
            stderr_lines = []
            ticker_count = 0
            total_tickers = 0  # Will be estimated from data directory
            
            # Try to estimate total tickers from data directory
            try:
                data_dir = PROJECT_ROOT / "data"
                if data_dir.exists():
                    feature_files = list(data_dir.glob("*.parquet"))
                    total_tickers = len(feature_files)
            except Exception:
                pass
            
            def read_stdout():
                nonlocal ticker_count
                try:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            if process.poll() is not None:
                                break
                            time.sleep(0.1)
                            continue
                        stdout_lines.append(line)
                        
                        # Try to detect ticker processing or progress indicators
                        # Look for patterns like "Error processing {ticker}" or ticker names
                        if re.search(r'Error processing|processing|ticker', line, re.IGNORECASE):
                            # Increment ticker count when we see processing messages
                            ticker_count += 1
                except Exception:
                    pass
            
            def read_stderr():
                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            if process.poll() is not None:
                                break
                            time.sleep(0.1)
                            continue
                        stderr_lines.append(line)
                except Exception:
                    pass
            
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor progress
            if progress_callback:
                start_time = time.time()
                last_update = start_time
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    
                    # Update progress based on ticker count or elapsed time
                    if total_tickers > 0:
                        # Use ticker count if we have total
                        progress_callback(min(ticker_count, total_tickers), total_tickers, 
                                        f"Processing tickers... ({ticker_count}/{total_tickers})")
                    else:
                        # Use elapsed time as proxy (estimate 1 ticker per 2 seconds)
                        estimated_tickers = max(1, int(elapsed / 2))
                        if elapsed < 60:  # First minute, show time-based estimate
                            progress_callback(estimated_tickers, max(estimated_tickers * 2, 100),
                                            f"Running backtest... ({int(elapsed)}s)")
                        else:
                            # After a minute, show we're working
                            progress_callback(50, 100, f"Running backtest... ({int(elapsed)}s elapsed)")
                    
                    time.sleep(0.5)  # Check every 0.5 seconds
            
            # Wait for completion
            process.wait()
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            
            # Small delay to ensure all buffered output is captured
            time.sleep(0.2)
            
            # Get final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            if process.returncode == 0:
                # Final progress update
                if progress_callback:
                    if total_tickers > 0:
                        progress_callback(total_tickers, total_tickers, "Backtest completed")
                    else:
                        progress_callback(100, 100, "Backtest completed")
                
                message_suffix = ""
                # Apply entry filters post-run (UI-level filtering)
                if entry_filters and output_path:
                    try:
                        df = pd.read_csv(output_path)
                        if not df.empty:
                            original_len = len(df)
                            df_filtered = self._apply_entry_filters(df, entry_filters)
                            df_filtered.to_csv(output_path, index=False)
                            filtered_len = len(df_filtered)
                            if filtered_len < original_len:
                                message_suffix = f" (filters applied: {filtered_len}/{original_len} trades kept)"
                            else:
                                message_suffix = " (filters applied: no change)"
                        else:
                            message_suffix = " (filters applied: no trades)"
                    except Exception as e:
                        message_suffix = f" (filters could not be applied: {e})"
                
                # Parse output for summary
                message = self._parse_backtest_summary(stdout, stderr) + message_suffix
                return True, output_path, message
            else:
                error_msg = stderr if stderr else stdout if stdout else f"Process exited with code {process.returncode}"
                return False, None, f"Backtest failed: {error_msg}"
        except Exception as e:
            return False, None, f"Error: {str(e)}"
    
    def _parse_backtest_summary(self, stdout: str, stderr: str) -> str:
        """Parse backtest output for summary information."""
        output = stdout + stderr
        
        # Look for key metrics in the output
        metrics = {}
        
        # Try to extract total trades
        trades_match = re.search(r'Total Trades:\s*(\d+)', output, re.IGNORECASE)
        if trades_match:
            metrics['trades'] = int(trades_match.group(1))
        
        # Try to extract win rate
        winrate_match = re.search(r'Win Rate:\s*([\d.]+)%', output, re.IGNORECASE)
        if winrate_match:
            metrics['win_rate'] = float(winrate_match.group(1))
        
        # Try to extract total P&L
        pnl_match = re.search(r'Total P&L:\s*\$?([\d,.-]+)', output, re.IGNORECASE)
        if pnl_match:
            metrics['pnl'] = pnl_match.group(1)
        
        # Build message
        if metrics:
            parts = []
            if 'trades' in metrics:
                parts.append(f"{metrics['trades']} trades")
            if 'win_rate' in metrics:
                parts.append(f"{metrics['win_rate']:.1f}% win rate")
            if 'pnl' in metrics:
                parts.append(f"${metrics['pnl']} P&L")
            
            if parts:
                return f"Backtest completed successfully ({', '.join(parts)})"
        
        return "Backtest completed successfully"

    def _apply_entry_filters(self, df: pd.DataFrame, filters: list) -> pd.DataFrame:
        """Apply entry filters to trades DataFrame."""
        filtered = df.copy()
        for feature, op, value in filters:
            if feature not in filtered.columns:
                continue
            try:
                if op == ">":
                    mask = filtered[feature] > value
                elif op == ">=":
                    mask = filtered[feature] >= value
                elif op == "<":
                    mask = filtered[feature] < value
                elif op == "<=":
                    mask = filtered[feature] <= value
                else:
                    continue
                filtered = filtered[mask]
            except Exception:
                # Skip invalid filter
                continue
        return filtered

