#!/usr/bin/env python3
"""
inspect_parquet_gui.py

GUI utility to inspect Parquet files, display summary information, and
export to CSV with automatic opening.

This is a standalone GUI application that:
  1. Lets you select a Parquet file via file dialog
  2. Displays summary information in the GUI
  3. Saves the DataFrame to CSV
  4. Automatically opens the CSV file

Usage:
    python src/inspect_parquet_gui.py
    
Or double-click the script to run it.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
import pandas as pd
import sys
import os
import subprocess
import platform

# Default export directory
DEFAULT_EXPORT_DIR = Path(__file__).parent.parent / "outputs" / "inspections"


class ParquetInspectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parquet File Inspector")
        self.root.geometry("900x700")
        
        self.df = None
        self.current_file = None
        self.export_dir = DEFAULT_EXPORT_DIR
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Top frame for file selection
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="Parquet File:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(top_frame, text="No file selected", fg="gray", anchor="w")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Button(top_frame, text="Select File", command=self.select_file, 
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side=tk.RIGHT, padx=5)
        
        # Middle frame for summary display
        mid_frame = tk.Frame(self.root, padx=10, pady=5)
        mid_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(mid_frame, text="File Summary:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.summary_text = scrolledtext.ScrolledText(mid_frame, wrap=tk.WORD, 
                                                      font=("Consolas", 9),
                                                      bg="#f5f5f5")
        self.summary_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Bottom frame for actions
        bottom_frame = tk.Frame(self.root, padx=10, pady=10)
        bottom_frame.pack(fill=tk.X)
        
        self.export_button = tk.Button(bottom_frame, text="Save as CSV and Open", 
                                      command=self.export_and_open,
                                      bg="#2196F3", fg="white", 
                                      font=("Arial", 10, "bold"),
                                      state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(bottom_frame, text="Clear", command=self.clear_display,
                 bg="#f44336", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(bottom_frame, text="Exit", command=self.root.quit,
                 font=("Arial", 10)).pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready - Select a Parquet file to begin", 
                                    bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_file(self):
        """Open file dialog to select a Parquet file."""
        file_path = filedialog.askopenfilename(
            title="Select Parquet File",
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")],
            initialdir=str(Path(__file__).parent.parent)
        )
        
        if file_path:
            self.current_file = Path(file_path)
            self.load_and_display()
    
    def load_and_display(self):
        """Load the Parquet file and display summary information."""
        try:
            self.status_label.config(text="Loading file...")
            self.root.update()
            
            # Load the Parquet file
            self.df = pd.read_parquet(self.current_file)
            
            # Update file label
            self.file_label.config(text=str(self.current_file.name), fg="black")
            
            # Generate summary
            summary = self.generate_summary()
            
            # Display summary
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, summary)
            
            # Enable export button
            self.export_button.config(state=tk.NORMAL)
            
            self.status_label.config(text=f"Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Parquet file:\n{str(e)}")
            self.status_label.config(text="Error loading file")
            self.export_button.config(state=tk.DISABLED)
    
    def generate_summary(self):
        """Generate summary text from the DataFrame."""
        if self.df is None:
            return "No data loaded."
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"File: {self.current_file.name}")
        lines.append("=" * 80)
        lines.append("")
        
        # Basic info
        lines.append(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        lines.append(f"Index name: {self.df.index.name or '(unnamed)'}")
        
        # Date range if index is datetime
        if isinstance(self.df.index, pd.DatetimeIndex):
            lines.append(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        elif len(self.df.index) > 0:
            lines.append(f"Index range: {self.df.index.min()} to {self.df.index.max()}")
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("Column Data Types:")
        lines.append("-" * 80)
        for col, dtype in self.df.dtypes.items():
            lines.append(f"  {col}: {dtype}")
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("Null Counts per Column:")
        lines.append("-" * 80)
        null_counts = self.df.isna().sum()
        for col, count in null_counts.items():
            pct = (count / len(self.df)) * 100 if len(self.df) > 0 else 0
            lines.append(f"  {col}: {count} ({pct:.1f}%)")
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("First 5 Rows:")
        lines.append("-" * 80)
        lines.append(str(self.df.head().to_string()))
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("Last 5 Rows:")
        lines.append("-" * 80)
        lines.append(str(self.df.tail().to_string()))
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_and_open(self):
        """Export DataFrame to CSV and open it."""
        if self.df is None:
            messagebox.showwarning("Warning", "No data loaded. Please select a file first.")
            return
        
        try:
            # Ensure export directory exists
            self.export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            out_csv = self.export_dir / f"{self.current_file.stem}.csv"
            
            self.status_label.config(text="Saving CSV...")
            self.root.update()
            
            # Save to CSV
            self.df.to_csv(out_csv, index=True)
            
            self.status_label.config(text=f"Saved to: {out_csv}")
            
            # Open the CSV file
            self.open_file(out_csv)
            
            messagebox.showinfo("Success", f"CSV saved and opened:\n{out_csv}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")
            self.status_label.config(text="Error saving file")
    
    def open_file(self, file_path):
        """Open a file using the system's default application."""
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', file_path])
            else:  # Linux
                subprocess.run(['xdg-open', file_path])
        except Exception as e:
            print(f"Warning: Could not open file automatically: {e}")
            messagebox.showinfo("File Saved", f"File saved to:\n{file_path}\n\nPlease open manually.")
    
    def clear_display(self):
        """Clear the display and reset state."""
        self.df = None
        self.current_file = None
        self.file_label.config(text="No file selected", fg="gray")
        self.summary_text.delete(1.0, tk.END)
        self.export_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready - Select a Parquet file to begin")


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = ParquetInspectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

