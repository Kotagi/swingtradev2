"""
Service layer that wraps CLI functions for use in the GUI.
This provides a clean interface between the GUI and the existing CLI code.
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any
import pandas as pd
import numpy as np
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

# Import SHAP service (optional - may not be available)
try:
    from src.shap_service import SHAPService
    HAS_SHAP_SERVICE = True
except ImportError:
    HAS_SHAP_SERVICE = False
    SHAPService = None


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
            str(scripts_dir / "feature_pipeline.py")
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
                message = self._parse_feature_summary(stdout, stderr)
                return True, message
            else:
                error_msg = stderr if stderr else stdout if stdout else f"Process exited with code {process.returncode}"
                return False, f"Feature building failed: {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _parse_feature_summary(self, stdout: str, stderr: str) -> str:
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
            message = "Features built successfully"
        
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
        shap: bool = False,
        imbalance_multiplier: float = 1.0,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        val_end: Optional[str] = None,
        horizon: Optional[int] = None,
        label_col: Optional[str] = None,
        return_threshold: Optional[float] = None,
        feature_set: Optional[str] = None,
        model_output: Optional[str] = None,
        progress_callback=None
    ) -> Tuple[bool, str, Dict]:
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
            return False, f"Training script not found: {train_script}", {}
        
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
        if shap:
            cmd.append("--shap")
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
        if return_threshold is not None:
            cmd.extend(["--return-threshold", str(return_threshold)])
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
                return False, f"Training failed to start: {error_msg}", {}
            
            # Wait for process to complete
            # Use a simple wait with periodic checks for progress updates
            start_time = time.time()
            last_progress_update = start_time
            
            while process.poll() is None:
                # Update progress periodically even if we're not detecting stages
                elapsed = time.time() - start_time
                if elapsed > 3600:  # 1 hour timeout
                    process.kill()
                    return False, "Training timed out after 1 hour", {}
                
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
                
                # Parse output to get model info and metrics
                message, metrics_dict = self._parse_training_summary(stdout, stderr)
                return True, message, metrics_dict
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
                return False, f"Training failed (exit code {returncode}): {error_msg}", {}
        except Exception as e:
            return False, f"Error: {str(e)}", {}
    
    def _parse_training_summary(self, stdout: str, stderr: str) -> Tuple[str, Dict]:
        """
        Parse training output to extract summary information and metrics.
        
        Returns:
            Tuple of (message: str, metrics_dict: Dict)
            metrics_dict contains: model_path, training_time, test_metrics, validation_metrics
        """
        import re
        
        # Combine stdout and stderr
        output = stdout + stderr
        
        metrics_dict = {}
        
        # Look for model file path
        model_path_match = re.search(r'Model saved to:\s*(.+)', output, re.IGNORECASE)
        model_path = model_path_match.group(1).strip() if model_path_match else None
        if model_path:
            metrics_dict['model_path'] = model_path
        
        # Look for training time
        time_match = re.search(r'Total training time:\s*([\d.]+)\s*seconds', output, re.IGNORECASE)
        training_time = time_match.group(1) if time_match else None
        if training_time:
            metrics_dict['training_time'] = float(training_time)
        
        # Look for feature count
        feature_count_match = re.search(r'Number of features:\s*(\d+)', output, re.IGNORECASE)
        if feature_count_match:
            metrics_dict['feature_count'] = int(feature_count_match.group(1))
        
        # Look for label column (used to extract horizon)
        # Patterns: "Using specified label column: label_30d"
        #           "Using label column from horizon: label_30d"
        #           "Auto-detected label column: label_30d"
        label_col_match = re.search(
            r'(?:Using (?:specified )?label column|Using label column from horizon|Auto-detected label column):\s*(label_\d+d)',
            output,
            re.IGNORECASE
        )
        if label_col_match:
            label_col = label_col_match.group(1)
            metrics_dict['label_col'] = label_col
            # Extract horizon from label_col (e.g., "label_30d" -> 30)
            horizon_match = re.search(r'label_(\d+)d', label_col, re.IGNORECASE)
            if horizon_match:
                metrics_dict['horizon'] = int(horizon_match.group(1))
        
        # Parse TEST SET metrics
        test_metrics = {}
        test_section_match = re.search(r'=== TEST SET METRICS ===(.*?)(?===|$)', output, re.IGNORECASE | re.DOTALL)
        if test_section_match:
            test_section = test_section_match.group(1)
            
            # Extract metrics using regex
            auc_match = re.search(r'ROC AUC:\s*([\d.]+)', test_section, re.IGNORECASE)
            if auc_match:
                test_metrics['roc_auc'] = float(auc_match.group(1))
            
            ap_match = re.search(r'Average Precision \(AP\):\s*([\d.]+)', test_section, re.IGNORECASE)
            if ap_match:
                test_metrics['average_precision'] = float(ap_match.group(1))
            
            f1_match = re.search(r'F1 Score:\s*([\d.]+)', test_section, re.IGNORECASE)
            if f1_match:
                test_metrics['f1_score'] = float(f1_match.group(1))
            
            precision_match = re.search(r'Precision \(Positive\):\s*([\d.]+)', test_section, re.IGNORECASE)
            if precision_match:
                test_metrics['precision'] = float(precision_match.group(1))
            
            recall_match = re.search(r'Recall \(Sensitivity\):\s*([\d.]+)', test_section, re.IGNORECASE)
            if recall_match:
                test_metrics['recall'] = float(recall_match.group(1))
            
            accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', test_section, re.IGNORECASE)
            if accuracy_match:
                test_metrics['accuracy'] = float(accuracy_match.group(1))
            
            # Confusion matrix
            cm_match = re.search(r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\]\]', test_section, re.IGNORECASE)
            if cm_match:
                test_metrics['confusion_matrix'] = [
                    [int(cm_match.group(1)), int(cm_match.group(2))],
                    [int(cm_match.group(3)), int(cm_match.group(4))]
                ]
        
        if test_metrics:
            metrics_dict['test_metrics'] = test_metrics
        
        # Parse VALIDATION SET metrics (if available)
        val_metrics = {}
        val_section_match = re.search(r'=== VALIDATION SET METRICS ===(.*?)(?===|$)', output, re.IGNORECASE | re.DOTALL)
        if val_section_match:
            val_section = val_section_match.group(1)
            
            auc_match = re.search(r'ROC AUC:\s*([\d.]+)', val_section, re.IGNORECASE)
            if auc_match:
                val_metrics['roc_auc'] = float(auc_match.group(1))
            
            ap_match = re.search(r'Average Precision \(AP\):\s*([\d.]+)', val_section, re.IGNORECASE)
            if ap_match:
                val_metrics['average_precision'] = float(ap_match.group(1))
            
            f1_match = re.search(r'F1 Score:\s*([\d.]+)', val_section, re.IGNORECASE)
            if f1_match:
                val_metrics['f1_score'] = float(f1_match.group(1))
            
            precision_match = re.search(r'Precision \(Positive\):\s*([\d.]+)', val_section, re.IGNORECASE)
            if precision_match:
                val_metrics['precision'] = float(precision_match.group(1))
            
            recall_match = re.search(r'Recall \(Sensitivity\):\s*([\d.]+)', val_section, re.IGNORECASE)
            if recall_match:
                val_metrics['recall'] = float(recall_match.group(1))
            
            accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', val_section, re.IGNORECASE)
            if accuracy_match:
                val_metrics['accuracy'] = float(accuracy_match.group(1))
        
        if val_metrics:
            metrics_dict['validation_metrics'] = val_metrics
        
        # Parse SHAP artifacts path if SHAP was computed
        shap_section_match = re.search(r'=== COMPUTING SHAP EXPLANATIONS ===(.*?)(?===|$)', output, re.IGNORECASE | re.DOTALL)
        if shap_section_match:
            shap_section = shap_section_match.group(1)
            
            # Extract SHAP artifacts path (matches "Artifacts saved to <path>")
            shap_path_match = re.search(r'Artifacts saved to\s+(.+)', shap_section, re.IGNORECASE)
            if shap_path_match:
                shap_path = shap_path_match.group(1).strip()
                metrics_dict['shap_artifacts_path'] = shap_path
            
            # Extract SHAP metadata
            data_split_match = re.search(r'Data split used:\s*(\w+)', shap_section, re.IGNORECASE)
            sample_size_match = re.search(r'Samples computed:\s*(\d+)', shap_section, re.IGNORECASE)
            
            shap_metadata = {}
            if data_split_match:
                shap_metadata['data_split'] = data_split_match.group(1).strip()
            if sample_size_match:
                shap_metadata['sample_size'] = int(sample_size_match.group(1))
            
            if shap_metadata:
                metrics_dict['shap_metadata'] = shap_metadata
        
        # Build message
        message = "Model training completed successfully"
        if model_path:
            message += f" - Model saved to: {Path(model_path).name}"
        if training_time:
            minutes = float(training_time) / 60
            message += f" ({minutes:.1f} minutes)"
        if test_metrics.get('roc_auc'):
            message += f" - Test ROC AUC: {test_metrics['roc_auc']:.4f}"
        if metrics_dict.get('shap_artifacts_path'):
            message += " - SHAP explanations computed"
        
        return message, metrics_dict


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


class StopLossAnalysisService:
    """Service for stop-loss analysis functionality."""
    
    def __init__(self):
        """Initialize the stop-loss analysis service."""
        self.data_dir = PROJECT_ROOT / "data" / "features_labeled"
        self.presets_dir = PROJECT_ROOT / "data" / "filter_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_stop_losses(
        self,
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame,
        effect_size_threshold: float = 0.3,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Analyze stop-loss trades from backtest data.
        
        Args:
            trades_df: DataFrame with trade data
            features_df: DataFrame with feature values at entry time
            effect_size_threshold: Minimum effect size for recommendations (default: 0.3)
            progress_callback: Callback(completed, total, message) for progress updates
            
        Returns:
            Analysis results dictionary with:
            - stop_loss_count, winning_count, target_count, total_trades
            - stop_loss_rate, immediate_stop_count, immediate_stop_rate
            - feature_comparisons (list of dicts)
            - recommendations (list of dicts)
            - immediate_stop_recommendations (list of dicts)
            - timing_analysis (dict)
            - holding_period_stats (dict)
            - return_stats (dict)
        """
        if progress_callback:
            progress_callback(0, 100, "Starting analysis...")
        
        results = {
            'stop_loss_count': 0,
            'winning_count': 0,
            'target_count': 0,
            'total_trades': len(trades_df),
            'stop_loss_rate': 0.0,
            'immediate_stop_count': 0,
            'immediate_stop_rate': 0.0,
            'feature_comparisons': [],
            'recommendations': [],
            'immediate_stop_recommendations': [],
            'timing_analysis': {},
            'holding_period_stats': {},
            'return_stats': {}
        }
        
        if trades_df.empty:
            return results
        
        # Separate trades by exit reason
        if progress_callback:
            progress_callback(10, 100, "Categorizing trades...")
        
        # Handle exit_reason - check both trades_df and features_df
        exit_reason_col = None
        if 'exit_reason' in trades_df.columns:
            exit_reason_col = trades_df['exit_reason']
        elif 'exit_reason' in features_df.columns:
            exit_reason_col = features_df['exit_reason']
        
        if exit_reason_col is not None:
            stop_loss_mask = exit_reason_col == 'stop_loss'
            target_mask = exit_reason_col == 'target_reached'
        else:
            # Fallback: use return to identify winners
            return_col = trades_df.get('return', features_df.get('return', pd.Series()))
            stop_loss_mask = return_col < 0
            target_mask = pd.Series([False] * len(trades_df))
        
        # Get return column
        return_col = trades_df.get('return', features_df.get('return', pd.Series()))
        if return_col.empty and 'return' in features_df.columns:
            return_col = features_df['return']
        
        winning_mask = return_col > 0
        
        # Count trades
        results['stop_loss_count'] = stop_loss_mask.sum() if hasattr(stop_loss_mask, 'sum') else len([x for x in stop_loss_mask if x])
        results['winning_count'] = winning_mask.sum() if hasattr(winning_mask, 'sum') else len([x for x in winning_mask if x])
        results['target_count'] = target_mask.sum() if hasattr(target_mask, 'sum') else len([x for x in target_mask if x])
        results['stop_loss_rate'] = results['stop_loss_count'] / results['total_trades'] if results['total_trades'] > 0 else 0.0
        
        # Get stop-loss trades for further analysis
        if results['stop_loss_count'] > 0:
            if 'exit_reason' in features_df.columns:
                stop_loss_features = features_df[features_df['exit_reason'] == 'stop_loss'].copy()
            elif 'exit_reason' in trades_df.columns:
                stop_loss_indices = trades_df[trades_df['exit_reason'] == 'stop_loss'].index
                stop_loss_features = features_df[features_df.index.isin(stop_loss_indices)].copy()
            else:
                # Use return < 0
                stop_loss_features = features_df[features_df.get('return', 0) < 0].copy()
            
            # Detect immediate stops (≤1 day)
            if 'holding_days' in stop_loss_features.columns:
                immediate_mask = stop_loss_features['holding_days'] <= 1
                results['immediate_stop_count'] = immediate_mask.sum()
                results['immediate_stop_rate'] = results['immediate_stop_count'] / results['stop_loss_count'] if results['stop_loss_count'] > 0 else 0.0
        else:
            stop_loss_features = pd.DataFrame()
        
        # Feature comparison
        if progress_callback:
            progress_callback(30, 100, "Comparing features...")
        
        if not features_df.empty and results['stop_loss_count'] > 0 and results['winning_count'] > 0:
            # Get winner features
            if 'exit_reason' in features_df.columns:
                winner_features = features_df[features_df['exit_reason'] != 'stop_loss'].copy()
                winner_features = winner_features[winner_features.get('return', 0) > 0]
            else:
                winner_features = features_df[features_df.get('return', 0) > 0].copy()
            
            if not stop_loss_features.empty and not winner_features.empty:
                # Get feature columns (exclude metadata)
                exclude_cols = {'ticker', 'entry_date', 'exit_reason', 'return', 'pnl', 'holding_days'}
                feature_cols = [col for col in features_df.columns if col not in exclude_cols]
                
                comparisons = []
                for i, feat in enumerate(feature_cols):
                    if progress_callback and i % 10 == 0:
                        progress_callback(30 + int(50 * i / len(feature_cols)), 100, f"Analyzing feature {i+1}/{len(feature_cols)}...")
                    
                    try:
                        sl_values = stop_loss_features[feat].dropna()
                        win_values = winner_features[feat].dropna()
                        
                        if len(sl_values) > 0 and len(win_values) > 0:
                            sl_mean = sl_values.mean()
                            win_mean = win_values.mean()
                            sl_std = sl_values.std()
                            win_std = win_values.std()
                            
                            if pd.notna(sl_mean) and pd.notna(win_mean) and sl_std > 0:
                                # Calculate Cohen's d (effect size)
                                pooled_std = np.sqrt((sl_std**2 + win_std**2) / 2)
                                if pooled_std > 0:
                                    cohens_d = (sl_mean - win_mean) / pooled_std
                                    abs_effect = abs(cohens_d)
                                    
                                    comparisons.append({
                                        'feature': feat,
                                        'stop_loss_mean': float(sl_mean),
                                        'winner_mean': float(win_mean),
                                        'difference': float(sl_mean - win_mean),
                                        'cohens_d': float(cohens_d),
                                        'abs_effect': float(abs_effect)
                                    })
                    except Exception:
                        continue
                
                # Sort by absolute effect size
                if comparisons:
                    comparisons_df = pd.DataFrame(comparisons)
                    comparisons_df = comparisons_df.sort_values('abs_effect', ascending=False)
                    results['feature_comparisons'] = comparisons_df.to_dict('records')
        
        # Generate recommendations
        if progress_callback:
            progress_callback(80, 100, "Generating recommendations...")
        
        if results['feature_comparisons']:
            recommendations = []
            for comp in results['feature_comparisons']:
                if comp['abs_effect'] >= effect_size_threshold:
                    feat = comp['feature']
                    sl_val = comp['stop_loss_mean']
                    win_val = comp['winner_mean']
                    cohens_d = comp['cohens_d']
                    
                    # Determine operator and value
                    if sl_val > win_val:
                        # Stop-loss trades have higher values - filter to exclude high values
                        operator = "<"
                        # Use a value between winner mean and stop-loss mean
                        threshold_value = win_val + (sl_val - win_val) * 0.3  # 30% toward winner mean
                    else:
                        # Stop-loss trades have lower values - filter to exclude low values
                        operator = ">"
                        # Use a value between stop-loss mean and winner mean
                        threshold_value = sl_val + (win_val - sl_val) * 0.3  # 30% toward stop-loss mean
                    
                    # Categorize effect size
                    abs_effect = comp['abs_effect']
                    if abs_effect > 0.5:
                        category = "strong"
                    elif abs_effect >= 0.3:
                        category = "moderate"
                    else:
                        category = "weak"
                    
                    recommendations.append({
                        'feature': feat,
                        'operator': operator,
                        'value': float(threshold_value),
                        'effect_size': float(abs_effect),
                        'cohens_d': float(cohens_d),
                        'category': category,
                        'description': f"Filter: {feat} {operator} {threshold_value:.4f} (effect size: {abs_effect:.3f})"
                    })
            
            results['recommendations'] = recommendations
        
        # Timing analysis
        if progress_callback:
            progress_callback(85, 100, "Analyzing timing patterns...")
        
        if results['stop_loss_count'] > 0 and not stop_loss_features.empty:
            # Get entry dates
            entry_dates = None
            if 'entry_date' in stop_loss_features.columns:
                entry_dates = pd.to_datetime(stop_loss_features['entry_date'])
            elif isinstance(stop_loss_features.index, pd.DatetimeIndex):
                entry_dates = stop_loss_features.index
            
            if entry_dates is not None:
                # Handle both Series and DatetimeIndex
                if isinstance(entry_dates, pd.Series):
                    day_of_week = entry_dates.dt.dayofweek
                    month = entry_dates.dt.month
                else:
                    # DatetimeIndex
                    day_of_week = entry_dates.dayofweek
                    month = entry_dates.month
                
                day_of_week_counts = day_of_week.value_counts().sort_index().to_dict()
                month_counts = month.value_counts().sort_index().to_dict()
                
                results['timing_analysis'] = {
                    'day_of_week': {int(k): int(v) for k, v in day_of_week_counts.items()},
                    'month': {int(k): int(v) for k, v in month_counts.items()}
                }
        
        # Holding period stats
        if progress_callback:
            progress_callback(90, 100, "Calculating holding period statistics...")
        
        if results['stop_loss_count'] > 0 and not stop_loss_features.empty and 'holding_days' in stop_loss_features.columns:
            holding_days = stop_loss_features['holding_days'].dropna()
            if len(holding_days) > 0:
                results['holding_period_stats'] = {
                    'mean': float(holding_days.mean()),
                    'median': float(holding_days.median()),
                    'min': float(holding_days.min()),
                    'max': float(holding_days.max()),
                    'distribution': {}  # Will be populated if needed for charts
                }
        
        # Return stats
        if progress_callback:
            progress_callback(95, 100, "Calculating return statistics...")
        
        if results['stop_loss_count'] > 0 and not stop_loss_features.empty and 'return' in stop_loss_features.columns:
            returns = stop_loss_features['return'].dropna()
            if len(returns) > 0:
                results['return_stats'] = {
                    'mean': float(returns.mean()),
                    'median': float(returns.median()),
                    'min': float(returns.min()),
                    'max': float(returns.max()),
                    'std_dev': float(returns.std())
                }
        
        if progress_callback:
            progress_callback(100, 100, "Analysis complete!")
        
        return results
    
    def get_entry_features(
        self,
        trades: pd.DataFrame,
        data_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Extract feature values at entry time for each trade.
        
        Args:
            trades: DataFrame with trade data (must have 'ticker' and entry date info)
            data_dir: Directory containing feature parquet files (default: data/features_labeled)
            progress_callback: Callback(completed, total, message) for progress updates
            
        Returns:
            DataFrame with trades + feature values at entry time
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        if not data_dir.exists():
            return pd.DataFrame()
        
        all_features = []
        total_trades = len(trades)
        skipped_count = 0
        
        for idx, (trade_idx, trade) in enumerate(trades.iterrows(), 1):
            # Get ticker
            ticker = trade.get('ticker', 'UNKNOWN') if 'ticker' in trade.index else 'UNKNOWN'
            if ticker == 'UNKNOWN':
                skipped_count += 1
                if progress_callback:
                    progress_callback(idx, total_trades, f"Skipping trade {idx}/{total_trades} (no ticker)")
                continue
            
            # Get entry_date - check index first, then column
            if isinstance(trade_idx, pd.Timestamp):
                entry_date = trade_idx
            elif 'entry_date' in trade.index:
                entry_date = trade['entry_date']
            else:
                skipped_count += 1
                if progress_callback:
                    progress_callback(idx, total_trades, f"Skipping trade {idx}/{total_trades} (no entry date)")
                continue
            
            if pd.isna(entry_date):
                skipped_count += 1
                if progress_callback:
                    progress_callback(idx, total_trades, f"Skipping trade {idx}/{total_trades} (invalid entry date)")
                continue
            
            # Load feature data for this ticker
            ticker_file = data_dir / f"{ticker}.parquet"
            if not ticker_file.exists():
                skipped_count += 1
                if progress_callback:
                    progress_callback(idx, total_trades, f"Skipping {ticker} (feature file not found)")
                continue
            
            try:
                df = pd.read_parquet(ticker_file).sort_index()
                
                # Find the row closest to entry date
                if isinstance(df.index, pd.DatetimeIndex):
                    entry_date_dt = pd.to_datetime(entry_date)
                    if entry_date_dt in df.index:
                        entry_row = df.loc[entry_date_dt]
                    else:
                        # Get closest date before entry
                        before_dates = df.index[df.index <= entry_date_dt]
                        if len(before_dates) > 0:
                            entry_row = df.loc[before_dates[-1]]
                        else:
                            skipped_count += 1
                            if progress_callback:
                                progress_callback(idx, total_trades, f"Skipping {ticker} (no data before entry date)")
                            continue
                else:
                    skipped_count += 1
                    if progress_callback:
                        progress_callback(idx, total_trades, f"Skipping {ticker} (invalid index type)")
                    continue
                
                # Extract features (exclude OHLCV, volume, labels)
                exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'adj close', 
                               'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
                exclude_cols.update([col for col in df.columns if col.startswith('label_')])
                
                features = {col: entry_row[col] for col in df.columns if col not in exclude_cols}
                features['ticker'] = ticker
                features['entry_date'] = entry_date
                
                # Get trade info
                features['exit_reason'] = trade.get('exit_reason', 'unknown')
                features['return'] = trade.get('return', 0)
                features['pnl'] = trade.get('pnl', 0)
                features['holding_days'] = trade.get('holding_days', 0)
                
                all_features.append(features)
                
                if progress_callback:
                    progress_callback(idx, total_trades, f"Processing {ticker} ({idx}/{total_trades})")
            
            except Exception as e:
                skipped_count += 1
                if progress_callback:
                    progress_callback(idx, total_trades, f"Error processing {ticker}: {str(e)[:50]}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(all_features)
        
        if progress_callback and skipped_count > 0:
            progress_callback(total_trades, total_trades, 
                            f"Completed: {len(all_features)} trades processed, {skipped_count} skipped")
        
        return result_df
    
    def calculate_impact(
        self,
        filters: List[Tuple[str, str, float]],
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate impact of applying filters to trades.
        
        Args:
            filters: List of (feature, operator, value) tuples
            trades_df: Original trades DataFrame
            features_df: Features at entry time DataFrame
            
        Returns:
            Dictionary with impact metrics:
            - stop_loss_excluded_pct: Percentage of stop-loss trades excluded
            - winner_excluded_pct: Percentage of winning trades excluded
            - estimated_new_sl_rate: Estimated new stop-loss rate
            - estimated_total_trades: Estimated total trades remaining
            - warnings: List of warning messages
        """
        if filters is None or len(filters) == 0:
            return {
                'stop_loss_excluded_pct': 0.0,
                'winner_excluded_pct': 0.0,
                'estimated_new_sl_rate': 0.0,
                'estimated_total_trades': len(trades_df),
                'warnings': []
            }
        
        if features_df.empty:
            return {
                'stop_loss_excluded_pct': 0.0,
                'winner_excluded_pct': 0.0,
                'estimated_new_sl_rate': 0.0,
                'estimated_total_trades': len(trades_df),
                'warnings': ['No feature data available for impact calculation']
            }
        
        # Start with all trades
        filtered_features = features_df.copy()
        
        # Apply each filter
        for feature, operator, value in filters:
            if feature not in filtered_features.columns:
                continue
            
            try:
                if operator == ">":
                    mask = filtered_features[feature] > value
                elif operator == ">=":
                    mask = filtered_features[feature] >= value
                elif operator == "<":
                    mask = filtered_features[feature] < value
                elif operator == "<=":
                    mask = filtered_features[feature] <= value
                else:
                    continue
                
                filtered_features = filtered_features[mask]
            except Exception:
                # Skip invalid filter
                continue
        
        # Count original trades by category
        total_original = len(features_df)
        
        # Identify stop-loss and winner trades
        if 'exit_reason' in features_df.columns:
            stop_loss_mask = features_df['exit_reason'] == 'stop_loss'
            winner_mask = features_df.get('return', pd.Series([0] * len(features_df))) > 0
        else:
            return_mask = features_df.get('return', pd.Series([0] * len(features_df)))
            stop_loss_mask = return_mask < 0
            winner_mask = return_mask > 0
        
        original_stop_loss_count = stop_loss_mask.sum() if hasattr(stop_loss_mask, 'sum') else len([x for x in stop_loss_mask if x])
        original_winner_count = winner_mask.sum() if hasattr(winner_mask, 'sum') else len([x for x in winner_mask if x])
        
        # Count filtered trades by category
        total_filtered = len(filtered_features)
        
        if 'exit_reason' in filtered_features.columns:
            filtered_stop_loss_mask = filtered_features['exit_reason'] == 'stop_loss'
            filtered_winner_mask = filtered_features.get('return', pd.Series([0] * len(filtered_features))) > 0
        else:
            filtered_return_mask = filtered_features.get('return', pd.Series([0] * len(filtered_features)))
            filtered_stop_loss_mask = filtered_return_mask < 0
            filtered_winner_mask = filtered_return_mask > 0
        
        filtered_stop_loss_count = filtered_stop_loss_mask.sum() if hasattr(filtered_stop_loss_mask, 'sum') else len([x for x in filtered_stop_loss_mask if x])
        filtered_winner_count = filtered_winner_mask.sum() if hasattr(filtered_winner_mask, 'sum') else len([x for x in filtered_winner_mask if x])
        
        # Calculate percentages
        stop_loss_excluded = original_stop_loss_count - filtered_stop_loss_count
        winner_excluded = original_winner_count - filtered_winner_count
        
        stop_loss_excluded_pct = (stop_loss_excluded / original_stop_loss_count * 100) if original_stop_loss_count > 0 else 0.0
        winner_excluded_pct = (winner_excluded / original_winner_count * 100) if original_winner_count > 0 else 0.0
        
        # Calculate new stop-loss rate
        estimated_new_sl_rate = (filtered_stop_loss_count / total_filtered * 100) if total_filtered > 0 else 0.0
        
        # Generate warnings
        warnings = []
        
        # Warning threshold: 50% of total trades excluded
        total_excluded_pct = ((total_original - total_filtered) / total_original * 100) if total_original > 0 else 0.0
        if total_excluded_pct >= 50.0:
            warnings.append(f"⚠️ Excluding {total_excluded_pct:.1f}% of total trades may significantly reduce trading opportunities")
        
        # Warning threshold: 20% of winners excluded
        if winner_excluded_pct >= 20.0:
            warnings.append(f"⚠️ Excluding {winner_excluded_pct:.1f}% of winning trades may reduce overall profitability")
        
        # Warning if no trades remain
        if total_filtered == 0:
            warnings.append("⚠️ All trades would be excluded by these filters")
        
        return {
            'stop_loss_excluded_pct': float(stop_loss_excluded_pct),
            'winner_excluded_pct': float(winner_excluded_pct),
            'estimated_new_sl_rate': float(estimated_new_sl_rate),
            'estimated_total_trades': int(total_filtered),
            'total_trades_before': int(total_original),
            'stop_loss_count_before': int(original_stop_loss_count),
            'stop_loss_count_after': int(filtered_stop_loss_count),
            'warnings': warnings
        }
    
    def save_preset(
        self,
        name: str,
        filters: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Save filter preset to file.
        
        Args:
            name: Preset name
            filters: List of filter dictionaries with feature, operator, value, effect_size, category
            metadata: Additional metadata (source_backtest, model_name, stop_loss_rate_before, etc.)
            
        Returns:
            Path to saved preset file
        """
        # Ensure presets directory exists
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize preset name for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        if not safe_name:
            safe_name = "preset"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = self.presets_dir / filename
        
        # Prepare preset data
        preset_data = {
            'name': name,
            'created_date': datetime.now().isoformat(),
            'filters': filters,
            **metadata  # Include all metadata (source_backtest, model_name, etc.)
        }
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        preset_data = self._convert_to_json_serializable(preset_data)
        
        # Save to JSON file
        import json
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)
        
        return str(filepath)
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            return obj
    
    def load_preset(self, preset_name: str) -> Dict:
        """
        Load filter preset from file.
        
        Args:
            preset_name: Name of preset to load (filename without .json extension, or full filename)
            
        Returns:
            Preset dictionary with filters and metadata
        """
        import json
        
        # Handle both filename with and without extension
        if not preset_name.endswith('.json'):
            preset_name = f"{preset_name}.json"
        
        filepath = self.presets_dir / preset_name
        
        if not filepath.exists():
            raise FileNotFoundError(f"Preset file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            preset_data = json.load(f)
        
        return preset_data
    
    def list_presets(self) -> List[Dict]:
        """
        List all available presets.
        
        Returns:
            List of preset metadata dictionaries (name, created_date, source_backtest, etc.)
        """
        import json
        
        if not self.presets_dir.exists():
            return []
        
        presets = []
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                
                # Extract metadata
                preset_info = {
                    'filename': preset_file.name,
                    'name': preset_data.get('name', preset_file.stem),
                    'created_date': preset_data.get('created_date', ''),
                    'source_backtest': preset_data.get('source_backtest'),
                    'model_name': preset_data.get('model_name'),
                    'stop_loss_rate_before': preset_data.get('stop_loss_rate_before', 0.0),
                    'stop_loss_rate_after': preset_data.get('stop_loss_rate_after', 0.0),
                    'filters_count': len(preset_data.get('filters', [])),
                    'total_trades_before': preset_data.get('total_trades_before', 0),
                    'total_trades_after': preset_data.get('total_trades_after', 0)
                }
                presets.append(preset_info)
            except Exception:
                # Skip corrupted files
                continue
        
        # Sort by created_date (newest first)
        presets.sort(key=lambda x: x.get('created_date', ''), reverse=True)
        
        return presets
    
    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a preset.
        
        Args:
            preset_name: Name of preset to delete (filename with or without .json extension)
            
        Returns:
            True if deleted, False if not found
        """
        # Handle both filename with and without extension
        if not preset_name.endswith('.json'):
            preset_name = f"{preset_name}.json"
        
        filepath = self.presets_dir / preset_name
        
        if not filepath.exists():
            return False
        
        try:
            filepath.unlink()
            return True
        except Exception:
            return False
    
    def rename_preset(self, old_filename: str, new_name: str) -> bool:
        """
        Rename a preset.
        
        Args:
            old_filename: Current preset filename (with or without .json extension)
            new_name: New preset name (will be sanitized for filename)
            
        Returns:
            True if renamed, False if not found
        """
        import json
        
        # Handle filename with or without extension
        if not old_filename.endswith('.json'):
            old_filename = f"{old_filename}.json"
        
        old_filepath = self.presets_dir / old_filename
        
        if not old_filepath.exists():
            return False
        
        # Load existing preset
        try:
            with open(old_filepath, 'r') as f:
                preset_data = json.load(f)
        except Exception:
            return False
        
        # Update name in preset data
        preset_data['name'] = new_name
        
        # Create new filename
        safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        if not safe_name:
            safe_name = "preset"
        
        # Keep same timestamp or use existing one
        timestamp = old_filename.split('_')[-1].replace('.json', '')
        if not timestamp or len(timestamp) != 15:  # Format: YYYYMMDD_HHMMSS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        new_filename = f"{safe_name}_{timestamp}.json"
        new_filepath = self.presets_dir / new_filename
        
        # Save with new name
        try:
            with open(new_filepath, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            # Delete old file
            old_filepath.unlink()
            return True
        except Exception:
            return False


class FilterEditorService:
    """Service for filter editor functionality."""
    
    def __init__(self):
        """Initialize the filter editor service."""
        self.data_dir = PROJECT_ROOT / "data" / "features_labeled"
        self.presets_dir = PROJECT_ROOT / "data" / "filter_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
    
    def get_feature_defaults(self, feature_name: str, features_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get default neutral point and increment for a feature.
        
        Args:
            feature_name: Name of the feature
            features_df: Optional DataFrame with feature values (for median calculation)
        
        Returns:
            Dictionary with 'neutral', 'increment', 'min', 'max', 'operator'
        """
        from gui.utils.feature_descriptions import FEATURE_DESCRIPTIONS
        
        defaults = {
            "neutral": 0.0,
            "increment": 0.1,
            "min": -1.0,
            "max": 1.0,
            "operator": ">"  # Default to greater than
        }
        
        # Get feature description if available
        feature_info = FEATURE_DESCRIPTIONS.get(feature_name, {})
        interpretation = feature_info.get("interpretation", "")
        
        # Determine feature type from interpretation or name
        if "[-1, +1]" in interpretation or "centered" in interpretation.lower():
            # Centered features (RSI, etc.)
            defaults["neutral"] = 0.0
            defaults["increment"] = 0.1
            defaults["min"] = -1.0
            defaults["max"] = 1.0
        elif "[0, 1]" in interpretation or "0-1" in interpretation.lower() or "normalized" in interpretation.lower():
            # Normalized 0-1 features
            defaults["neutral"] = 0.5
            defaults["increment"] = 0.05
            defaults["min"] = 0.0
            defaults["max"] = 1.0
        elif "[-0.2, 0.2]" in interpretation or "±20%" in interpretation:
            # Clipped return features
            defaults["neutral"] = 0.0
            defaults["increment"] = 0.05
            defaults["min"] = -0.2
            defaults["max"] = 0.2
        elif "[0.0, 0.2]" in interpretation or "0% to 20%" in interpretation:
            # Percentage-based (0-20%)
            defaults["neutral"] = 0.1
            defaults["increment"] = 0.01
            defaults["min"] = 0.0
            defaults["max"] = 0.2
        elif feature_name.startswith("price") and "log" not in feature_name:
            # Raw price (unbounded)
            if features_df is not None and feature_name in features_df.columns:
                defaults["neutral"] = float(features_df[feature_name].median())
                defaults["increment"] = max(1.0, defaults["neutral"] * 0.1)
                defaults["min"] = float(features_df[feature_name].quantile(0.01))
                defaults["max"] = float(features_df[feature_name].quantile(0.99))
            else:
                defaults["neutral"] = 10.0
                defaults["increment"] = 1.0
                defaults["min"] = 0.0
                defaults["max"] = 1000.0
        else:
            # Unbounded features - use median if available
            if features_df is not None and feature_name in features_df.columns:
                median_val = float(features_df[feature_name].median())
                std_val = float(features_df[feature_name].std())
                defaults["neutral"] = median_val
                defaults["increment"] = max(0.1, std_val * 0.1)
                defaults["min"] = float(features_df[feature_name].quantile(0.01))
                defaults["max"] = float(features_df[feature_name].quantile(0.99))
        
        return defaults
    
    def calculate_filter_impact(
        self,
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame,
        filters: List[Tuple[str, str, float]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Calculate the impact of applying filters to trades.
        
        Args:
            trades_df: DataFrame with trade data (must have 'return', 'pnl', 'exit_reason' columns)
            features_df: DataFrame with feature values at entry time (indexed by trade entry_date)
            filters: List of (feature, operator, value) tuples
            progress_callback: Optional callback(completed, total, message)
        
        Returns:
            Dictionary with impact metrics:
            - per_feature: Dict of impact per individual filter
            - combined: Combined impact of all filters
            - before_metrics: Metrics before filtering
            - after_metrics: Metrics after filtering
        """
        if progress_callback:
            progress_callback(0, len(filters) + 2, "Calculating baseline metrics...")
        
        # Calculate baseline metrics (before filtering)
        from src.enhanced_backtest import calculate_metrics
        
        # Ensure trades_df has required columns
        if "return" not in trades_df.columns:
            trades_df["return"] = (trades_df.get("exit_price", 0) / trades_df.get("entry_price", 1) - 1) if "exit_price" in trades_df.columns else 0.0
        if "pnl" not in trades_df.columns:
            trades_df["pnl"] = trades_df.get("return", 0) * trades_df.get("position_size", 1000)
        if "holding_days" not in trades_df.columns and "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
            trades_df["holding_days"] = (pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(trades_df["entry_date"])).dt.days
        
        before_metrics = calculate_metrics(trades_df, position_size=1000.0)
        
        # Identify winners and stop-losses
        winners = trades_df[trades_df["return"] > 0].copy()
        stop_losses = trades_df[
            (trades_df.get("exit_reason", "").str.contains("stop_loss", case=False, na=False)) |
            (trades_df.get("return", 0) < -0.05)  # Fallback: large negative return
        ].copy()
        
        # Calculate per-feature impact
        per_feature_impact = {}
        
        for idx, (feature, operator, value) in enumerate(filters):
            if progress_callback:
                progress_callback(idx + 1, len(filters) + 2, f"Calculating impact for {feature}...")
            
            if feature not in features_df.columns:
                per_feature_impact[feature] = {
                    "winners_excluded": 0,
                    "winners_excluded_pct": 0.0,
                    "stop_losses_excluded": 0,
                    "stop_losses_excluded_pct": 0.0,
                    "total_excluded": 0,
                    "total_excluded_pct": 0.0,
                    "error": f"Feature '{feature}' not found in data"
                }
                continue
            
            # Apply single filter
            feature_values = features_df[feature]
            
            if operator == ">":
                mask = feature_values > value
            elif operator == ">=":
                mask = feature_values >= value
            elif operator == "<":
                mask = feature_values < value
            elif operator == "<=":
                mask = feature_values <= value
            else:
                mask = pd.Series(True, index=features_df.index)
            
            # Get excluded trades (those that don't pass the filter)
            excluded_mask = ~mask
            
            # Match features_df index with trades_df
            # Features are indexed by (ticker, entry_date) or entry_date
            # Trades are in trades_df with entry_date column or index
            trade_mask = pd.Series(False, index=trades_df.index)
            
            if "entry_date" in trades_df.columns:
                # Match by entry_date
                entry_dates = pd.to_datetime(trades_df["entry_date"])
                if isinstance(features_df.index, pd.MultiIndex):
                    # MultiIndex (ticker, entry_date) - need to match both
                    if "ticker" in trades_df.columns:
                        for idx in trades_df.index:
                            ticker = trades_df.loc[idx, "ticker"]
                            entry_date = pd.to_datetime(trades_df.loc[idx, "entry_date"])
                            if (ticker, entry_date) in features_df.index:
                                feature_idx = (ticker, entry_date)
                                if feature_idx in excluded_mask.index:
                                    trade_mask.loc[idx] = excluded_mask.loc[feature_idx]
                elif isinstance(features_df.index, pd.DatetimeIndex):
                    # DatetimeIndex - match by entry_date
                    for idx in trades_df.index:
                        entry_date = pd.to_datetime(trades_df.loc[idx, "entry_date"])
                        if entry_date in features_df.index:
                            if entry_date in excluded_mask.index:
                                trade_mask.loc[idx] = excluded_mask.loc[entry_date]
                else:
                    # Integer index - assume same order
                    if len(excluded_mask) == len(trades_df):
                        trade_mask = pd.Series(excluded_mask.values, index=trades_df.index)
            else:
                # No entry_date column - try index matching
                if len(excluded_mask) == len(trades_df):
                    trade_mask = pd.Series(excluded_mask.values, index=trades_df.index)
            
            excluded_trades = trades_df[trade_mask]
            
            if not excluded_trades.empty:
                excluded_winners = excluded_trades[excluded_trades["return"] > 0]
                excluded_stop_losses = excluded_trades[
                    (excluded_trades.get("exit_reason", "").str.contains("stop_loss", case=False, na=False)) |
                    (excluded_trades.get("return", 0) < -0.05)
                ]
                
                per_feature_impact[feature] = {
                    "winners_excluded": len(excluded_winners),
                    "winners_excluded_pct": (len(excluded_winners) / len(winners) * 100) if len(winners) > 0 else 0.0,
                    "stop_losses_excluded": len(excluded_stop_losses),
                    "stop_losses_excluded_pct": (len(excluded_stop_losses) / len(stop_losses) * 100) if len(stop_losses) > 0 else 0.0,
                    "total_excluded": len(excluded_trades),
                    "total_excluded_pct": (len(excluded_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0.0
                }
            else:
                per_feature_impact[feature] = {
                    "winners_excluded": 0,
                    "winners_excluded_pct": 0.0,
                    "stop_losses_excluded": 0,
                    "stop_losses_excluded_pct": 0.0,
                    "total_excluded": 0,
                    "total_excluded_pct": 0.0
                }
        
        # Calculate combined impact
        if progress_callback:
            progress_callback(len(filters) + 1, len(filters) + 2, "Calculating combined impact...")
        
        # Apply all filters together
        combined_mask = pd.Series(True, index=features_df.index)
        
        for feature, operator, value in filters:
            if feature not in features_df.columns:
                continue
            
            feature_values = features_df[feature]
            
            if operator == ">":
                feature_mask = feature_values > value
            elif operator == ">=":
                feature_mask = feature_values >= value
            elif operator == "<":
                feature_mask = feature_values < value
            elif operator == "<=":
                feature_mask = feature_values <= value
            else:
                feature_mask = pd.Series(True, index=features_df.index)
            
            combined_mask = combined_mask & feature_mask
        
        # Get filtered trades - match features with trades
        trade_mask = pd.Series(True, index=trades_df.index)
        
        if "entry_date" in trades_df.columns:
            # Match by entry_date
            if isinstance(features_df.index, pd.MultiIndex):
                # MultiIndex (ticker, entry_date)
                if "ticker" in trades_df.columns:
                    for idx in trades_df.index:
                        ticker = trades_df.loc[idx, "ticker"]
                        entry_date = pd.to_datetime(trades_df.loc[idx, "entry_date"])
                        if (ticker, entry_date) in features_df.index:
                            feature_idx = (ticker, entry_date)
                            if feature_idx in combined_mask.index:
                                trade_mask.loc[idx] = combined_mask.loc[feature_idx]
                        else:
                            trade_mask.loc[idx] = False  # No feature data = exclude
            elif isinstance(features_df.index, pd.DatetimeIndex):
                # DatetimeIndex - match by entry_date
                for idx in trades_df.index:
                    entry_date = pd.to_datetime(trades_df.loc[idx, "entry_date"])
                    if entry_date in features_df.index:
                        if entry_date in combined_mask.index:
                            trade_mask.loc[idx] = combined_mask.loc[entry_date]
                    else:
                        trade_mask.loc[idx] = False  # No feature data = exclude
            else:
                # Integer index - assume same order
                if len(combined_mask) == len(trades_df):
                    trade_mask = pd.Series(combined_mask.values, index=trades_df.index)
        else:
            # No entry_date - try index matching
            if len(combined_mask) == len(trades_df):
                trade_mask = pd.Series(combined_mask.values, index=trades_df.index)
        
        filtered_trades = trades_df[trade_mask]
        
        # Calculate after metrics
        after_metrics = calculate_metrics(filtered_trades, position_size=1000.0) if not filtered_trades.empty else {
            "n_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "annual_return": 0.0
        }
        
        # Calculate excluded counts
        excluded_trades = trades_df[~trades_df.index.isin(filtered_trades.index)] if not filtered_trades.empty else trades_df
        excluded_winners = excluded_trades[excluded_trades["return"] > 0] if not excluded_trades.empty else pd.DataFrame()
        excluded_stop_losses = excluded_trades[
            (excluded_trades.get("exit_reason", "").str.contains("stop_loss", case=False, na=False)) |
            (excluded_trades.get("return", 0) < -0.05)
        ] if not excluded_trades.empty else pd.DataFrame()
        
        combined_impact = {
            "winners_excluded": len(excluded_winners),
            "winners_excluded_pct": (len(excluded_winners) / len(winners) * 100) if len(winners) > 0 else 0.0,
            "stop_losses_excluded": len(excluded_stop_losses),
            "stop_losses_excluded_pct": (len(excluded_stop_losses) / len(stop_losses) * 100) if len(stop_losses) > 0 else 0.0,
            "total_excluded": len(excluded_trades),
            "total_excluded_pct": (len(excluded_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0.0,
            "remaining_trades": len(filtered_trades),
            "remaining_trades_pct": (len(filtered_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0.0
        }
        
        if progress_callback:
            progress_callback(len(filters) + 2, len(filters) + 2, "Impact calculation complete!")
        
        return {
            "per_feature": per_feature_impact,
            "combined": combined_impact,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics
        }
    
    def save_filter_preset(
        self,
        name: str,
        filters: List[Tuple[str, str, float]],
        description: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save a filter preset.
        
        Args:
            name: Preset name
            filters: List of (feature, operator, value) tuples
            description: Optional description
            tags: Optional list of tags
            metadata: Optional additional metadata
        
        Returns:
            True if saved successfully
        """
        try:
            preset_data = {
                "name": name,
                "description": description or "",
                "tags": tags or [],
                "filters": [
                    {"feature": f, "operator": op, "value": float(v)}
                    for f, op, v in filters
                ],
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Convert to JSON-serializable
            preset_data = self._convert_to_json_serializable(preset_data)
            
            # Sanitize filename
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"{safe_name}.json"
            filepath = self.presets_dir / filename
            
            # Handle duplicates
            counter = 1
            while filepath.exists():
                filename = f"{safe_name}_{counter}.json"
                filepath = self.presets_dir / filename
                counter += 1
            
            with open(filepath, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving filter preset: {e}")
            return False
    
    def load_filter_preset(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load a filter preset.
        
        Args:
            filename: Preset filename (with or without .json extension)
        
        Returns:
            Preset data dictionary or None if not found
        """
        try:
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.presets_dir / filename
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                preset_data = json.load(f)
            
            # Convert filters back to tuples
            if "filters" in preset_data:
                preset_data["filters"] = [
                    (f["feature"], f["operator"], f["value"])
                    for f in preset_data["filters"]
                ]
            
            return preset_data
        except Exception as e:
            print(f"Error loading filter preset: {e}")
            return None
    
    def list_filter_presets(self) -> List[Dict[str, Any]]:
        """List all filter presets with metadata."""
        presets = []
        
        for filepath in self.presets_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    preset_data = json.load(f)
                
                preset_data["filename"] = filepath.name
                presets.append(preset_data)
            except Exception:
                continue
        
        # Sort by modified date (newest first)
        presets.sort(key=lambda x: x.get("modified", ""), reverse=True)
        return presets
    
    def delete_filter_preset(self, filename: str) -> bool:
        """Delete a filter preset."""
        try:
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.presets_dir / filename
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception:
            return False
    
    def rename_filter_preset(self, old_filename: str, new_name: str) -> bool:
        """Rename a filter preset."""
        try:
            if not old_filename.endswith('.json'):
                old_filename += '.json'
            
            old_filepath = self.presets_dir / old_filename
            
            if not old_filepath.exists():
                return False
            
            # Load old preset
            with open(old_filepath, 'r') as f:
                preset_data = json.load(f)
            
            # Update name
            preset_data["name"] = new_name
            preset_data["modified"] = datetime.now().isoformat()
            
            # Sanitize new filename
            safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            new_filename = f"{safe_name}.json"
            new_filepath = self.presets_dir / new_filename
            
            # Handle duplicates
            counter = 1
            while new_filepath.exists() and new_filepath != old_filepath:
                new_filename = f"{safe_name}_{counter}.json"
                new_filepath = self.presets_dir / new_filename
                counter += 1
            
            # Save new file
            with open(new_filepath, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            # Delete old file if different
            if new_filepath != old_filepath:
                old_filepath.unlink()
            
            return True
        except Exception:
            return False
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj


class SHAPService:
    """Service for SHAP explainability operations."""
    
    def __init__(self):
        """Initialize SHAP service."""
        try:
            from src.shap_service import SHAPService as _SHAPService
            self._shap_service = _SHAPService()
            self._available = True
        except ImportError:
            self._shap_service = None
            self._available = False
    
    def is_available(self) -> bool:
        """Check if SHAP service is available."""
        return self._available
    
    def load_artifacts(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load SHAP artifacts for a model.
        
        Args:
            model_id: Model identifier (typically the model filename without extension)
        
        Returns:
            Dictionary with SHAP artifacts, or None if not found
        """
        if not self._available:
            return None
        
        return self._shap_service.load_artifacts(model_id)
    
    def artifact_exists(self, model_id: str) -> bool:
        """Check if SHAP artifacts exist for a model."""
        if not self._available:
            return False
        
        return self._shap_service.artifact_exists(model_id)
    
    def get_model_id_from_path(self, model_path: str) -> str:
        """
        Extract model ID from model file path.
        
        Args:
            model_path: Path to model file
        
        Returns:
            Model ID (filename without extension)
        """
        return Path(model_path).stem
    
    def recompute_shap(
        self,
        model_path: str,
        model_registry_entry: Dict[str, Any],
        progress_callback=None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Recompute SHAP explanations for an existing model.
        
        Args:
            model_path: Path to the model .pkl file
            model_registry_entry: Model registry entry with parameters (horizon, return_threshold, feature_set, etc.)
            progress_callback: Optional function(stage, message) to call for progress updates
        
        Returns:
            Tuple of (success: bool, message: str, metadata: Optional[Dict])
        """
        if not self._available:
            return False, "SHAP library not available. Install with: pip install shap", {}
        
        try:
            import joblib
            import pandas as pd
            import numpy as np
            import yaml
            from pathlib import Path
            
            # Import utilities needed for data loading and preparation
            import sys
            PROJECT_ROOT = Path(__file__).parent.parent
            SRC_DIR = PROJECT_ROOT / "src"
            sys.path.insert(0, str(SRC_DIR))
            
            from utils.labeling import label_future_return
            
            # Try to import feature set manager
            try:
                from feature_set_manager import (
                    get_feature_set_data_path,
                    get_train_features_config_path,
                    DEFAULT_FEATURE_SET
                )
                HAS_FEATURE_SET_MANAGER = True
            except ImportError:
                HAS_FEATURE_SET_MANAGER = False
                DEFAULT_FEATURE_SET = "v1"
            
            if progress_callback:
                progress_callback(1, "Loading model...")
            
            # Load model
            model_data = joblib.load(model_path)
            model_id = Path(model_path).stem
            
            # Extract model from dict (models are saved as dict with "model", "features", "metadata")
            if isinstance(model_data, dict):
                model = model_data.get("model")
                if model is None:
                    # Fallback: maybe the whole thing is the model (old format)
                    model = model_data
            else:
                model = model_data
            
            if model is None:
                return False, "Could not extract model from file. Model file may be corrupted.", {}
            
            # Get parameters from registry
            params = model_registry_entry.get("parameters", {})
            training_info = model_registry_entry.get("training_info", {})
            
            # Get feature set, handling None case
            feature_set = params.get("feature_set") or training_info.get("feature_set")
            if not feature_set or feature_set == "None" or str(feature_set).lower() == "none":
                feature_set = DEFAULT_FEATURE_SET
            
            horizon = params.get("horizon") or training_info.get("horizon") or 30
            return_threshold = params.get("return_threshold") or training_info.get("return_threshold") or 0.05
            
            if progress_callback:
                progress_callback(2, f"Loading feature data (feature set: {feature_set})...")
            
            # Determine data directory based on feature set
            if HAS_FEATURE_SET_MANAGER:
                data_dir = get_feature_set_data_path(feature_set)
                train_cfg_path = get_train_features_config_path(feature_set)
            else:
                data_dir = PROJECT_ROOT / "data" / "features_labeled"
                train_cfg_path = PROJECT_ROOT / "config" / "train_features.yaml"
            
            if not data_dir.exists():
                return False, f"Feature data directory not found: {data_dir}", {}
            
            # Load feature data
            parts = []
            parquet_files = list(data_dir.glob("*.parquet"))
            if not parquet_files:
                return False, f"No feature files found in {data_dir}", {}
            
            for f in parquet_files:
                df = pd.read_parquet(f)
                df.index.name = "date"
                df["ticker"] = f.stem
                parts.append(df)
            
            df_all = pd.concat(parts, axis=0)
            
            if progress_callback:
                progress_callback(3, f"Calculating labels (horizon={horizon}d, threshold={return_threshold:.2%})...")
            
            # Find close and high columns (case-insensitive)
            close_col = None
            high_col = None
            for col in df_all.columns:
                if col.lower() in ['close', 'adj close']:
                    close_col = col
                if col.lower() == 'high':
                    high_col = col
            
            if not close_col:
                return False, "Could not find 'close' or 'adj close' column in data", {}
            if not high_col:
                return False, "Could not find 'high' column in data", {}
            
            # Ensure columns are numeric
            if df_all[close_col].dtype == 'object':
                df_all[close_col] = pd.to_numeric(df_all[close_col], errors='coerce')
            if df_all[high_col].dtype == 'object':
                df_all[high_col] = pd.to_numeric(df_all[high_col], errors='coerce')
            
            # Calculate labels
            label_col = "training_label"
            df_all = label_future_return(
                df_all,
                close_col=close_col,
                high_col=high_col,
                horizon=horizon,
                threshold=return_threshold,
                label_name=label_col
            )
            
            if progress_callback:
                progress_callback(4, "Preparing data...")
            
            # Prepare data (clean, drop NaNs, exclude raw prices and forward returns)
            df_clean = df_all.replace([np.inf, -np.inf], np.nan).dropna().copy()
            
            # Get feature names (exclude label, ticker, raw prices, forward returns)
            forward_return_cols = {"5d_return", "10d_return"}
            raw_price_cols = {"open", "high", "low", "close", "adj close"}
            all_feats = [c for c in df_clean.columns 
                        if c not in [label_col, "ticker"] 
                        and c.lower() not in forward_return_cols
                        and c.lower() not in raw_price_cols]
            
            # Load training config to get enabled features
            if not train_cfg_path.exists():
                return False, f"Training config not found: {train_cfg_path}", {}
            
            train_cfg = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8")) or {}
            flags = train_cfg.get("features", {})
            enabled_feats = {name for name, flag in flags.items() if flag == 1}
            feats = [f for f in all_feats if f in enabled_feats]
            
            if not feats:
                return False, "No features selected; check train_features.yaml", {}
            
            # Extract X and y
            X_all = df_clean[feats]
            y_all = df_clean[label_col]
            
            # Split data by date (use same logic as training)
            dates = pd.to_datetime(X_all.index)
            train_cutoff = pd.to_datetime("2022-12-31")
            val_cutoff = pd.to_datetime("2023-12-31")
            
            train_mask = dates <= train_cutoff
            val_mask = (dates > train_cutoff) & (dates <= val_cutoff)
            test_mask = dates > val_cutoff
            
            X_val = X_all[val_mask]
            y_val = y_all[val_mask]
            
            # Use validation set if available, otherwise test set
            if len(X_val) >= 100:
                shap_data = X_val
                shap_labels = y_val
                data_split_name = "validation"
            elif len(X_all[test_mask]) >= 100:
                shap_data = X_all[test_mask]
                shap_labels = y_all[test_mask]
                data_split_name = "test"
            else:
                # Fall back to training set if validation/test are too small
                shap_data = X_all[train_mask]
                shap_labels = y_all[train_mask]
                data_split_name = "training"
            
            if progress_callback:
                progress_callback(5, f"Computing SHAP values ({len(shap_data)} samples)...")
            
            # Compute SHAP
            result = self._shap_service.compute_shap(
                model=model,
                X_data=shap_data,
                y_data=shap_labels,
                features=feats,
                model_id=model_id,
                sample_size=1000,
                use_stratified=True,
                data_split=data_split_name
            )
            
            if result["success"]:
                if progress_callback:
                    progress_callback(6, "SHAP computation complete!")
                # Return metadata dict with artifacts_path included
                metadata = result.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                if result.get("artifacts_path"):
                    metadata["artifacts_path"] = str(result["artifacts_path"])
                return True, result["message"], metadata
            else:
                return False, result["message"], {}
                
        except Exception as e:
            import traceback
            error_msg = f"Error recomputing SHAP: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg, {}

