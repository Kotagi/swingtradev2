"""
Configuration Manager for GUI

Handles saving and loading presets for all tabs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "gui" / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages saving and loading of GUI configuration presets."""
    
    def __init__(self):
        self.config_dir = CONFIG_DIR
    
    def save_preset(self, preset_name: str, tab_name: str, config: Dict[str, Any]) -> bool:
        """
        Save a configuration preset for a specific tab.
        
        Args:
            preset_name: Name of the preset
            tab_name: Name of the tab (e.g., 'data', 'features', 'training', 'backtest', 'identify')
            config: Dictionary of configuration values
        
        Returns:
            True if successful, False otherwise
        """
        try:
            preset_file = self.config_dir / f"{tab_name}_{preset_name}.json"
            
            preset_data = {
                "preset_name": preset_name,
                "tab_name": tab_name,
                "config": config,
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat()
            }
            
            with open(preset_file, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving preset: {e}")
            return False
    
    def load_preset(self, preset_name: str, tab_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration preset for a specific tab.
        
        Args:
            preset_name: Name of the preset
            tab_name: Name of the tab
        
        Returns:
            Configuration dictionary or None if not found
        """
        try:
            preset_file = self.config_dir / f"{tab_name}_{preset_name}.json"
            
            if not preset_file.exists():
                return None
            
            with open(preset_file, 'r') as f:
                preset_data = json.load(f)
            
            return preset_data.get("config", {})
        except Exception as e:
            print(f"Error loading preset: {e}")
            return None
    
    def list_presets(self, tab_name: str) -> list:
        """
        List all available presets for a specific tab.
        
        Args:
            tab_name: Name of the tab
        
        Returns:
            List of preset names
        """
        try:
            pattern = f"{tab_name}_*.json"
            preset_files = list(self.config_dir.glob(pattern))
            
            presets = []
            for preset_file in preset_files:
                try:
                    with open(preset_file, 'r') as f:
                        preset_data = json.load(f)
                    presets.append({
                        "name": preset_data.get("preset_name", preset_file.stem),
                        "created": preset_data.get("created", ""),
                        "updated": preset_data.get("updated", "")
                    })
                except Exception:
                    # Skip invalid files
                    continue
            
            # Sort by updated date (most recent first)
            presets.sort(key=lambda x: x.get("updated", ""), reverse=True)
            return [p["name"] for p in presets]
        except Exception as e:
            print(f"Error listing presets: {e}")
            return []
    
    def delete_preset(self, preset_name: str, tab_name: str) -> bool:
        """
        Delete a configuration preset.
        
        Args:
            preset_name: Name of the preset
            tab_name: Name of the tab
        
        Returns:
            True if successful, False otherwise
        """
        try:
            preset_file = self.config_dir / f"{tab_name}_{preset_name}.json"
            
            if preset_file.exists():
                preset_file.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting preset: {e}")
            return False
    
    def get_preset_info(self, preset_name: str, tab_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a preset.
        
        Args:
            preset_name: Name of the preset
            tab_name: Name of the tab
        
        Returns:
            Dictionary with preset metadata or None
        """
        try:
            preset_file = self.config_dir / f"{tab_name}_{preset_name}.json"
            
            if not preset_file.exists():
                return None
            
            with open(preset_file, 'r') as f:
                preset_data = json.load(f)
            
            return {
                "name": preset_data.get("preset_name", preset_name),
                "tab": preset_data.get("tab_name", tab_name),
                "created": preset_data.get("created", ""),
                "updated": preset_data.get("updated", "")
            }
        except Exception as e:
            print(f"Error getting preset info: {e}")
            return None

