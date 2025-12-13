"""
Model Registry - Manages storage and retrieval of model metadata and performance metrics.

This module provides functionality to:
- Save model metadata after training
- Load and query model registry
- Compare models side-by-side
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ModelRegistry:
    """Manages model metadata storage and retrieval."""
    
    def __init__(self, registry_file: Optional[Path] = None):
        """
        Initialize the model registry.
        
        Args:
            registry_file: Path to JSON registry file. Defaults to models/models_registry.json
        """
        if registry_file is None:
            # Default to models directory
            self.registry_file = Path(__file__).parent.parent.parent / "models" / "models_registry.json"
        else:
            self.registry_file = Path(registry_file)
        
        # Ensure models directory exists
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start fresh
                return {"models": [], "version": "1.0"}
        return {"models": [], "version": "1.0"}
    
    def _save_registry(self):
        """Save registry to JSON file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2, default=str)
        except IOError as e:
            raise IOError(f"Failed to save model registry: {e}")
    
    def register_model(
        self,
        model_path: str,
        metrics: Dict[str, Any],
        parameters: Dict[str, Any],
        training_info: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_path: Path to the model file
            metrics: Dictionary of performance metrics (e.g., accuracy, roc_auc, etc.)
            parameters: Dictionary of training parameters (horizon, return_threshold, etc.)
            training_info: Additional training information (training_time, feature_count, etc.)
            name: Optional custom name for the model. If None, auto-generates from parameters.
        
        Returns:
            Model ID (unique identifier)
        """
        # Generate model ID from timestamp
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate name if not provided
        if name is None:
            horizon = parameters.get('horizon')
            threshold = parameters.get('return_threshold', 0.0)
            
            # Format horizon (handle None)
            if horizon is not None:
                horizon_str = f"{horizon}d"
            else:
                horizon_str = "?"
            
            # Format threshold (handle None)
            if threshold is not None and threshold > 0:
                threshold_str = f"{threshold:.0%}"
            else:
                threshold_str = "0%"
            
            name = f"XGBoost_{horizon_str}_{threshold_str}"
        
        # Prepare model entry
        model_entry = {
            "id": model_id,
            "name": name,
            "file_path": str(model_path),
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "parameters": parameters,
            "training_info": training_info or {}
        }
        
        # Add to registry
        self._registry["models"].append(model_entry)
        
        # Save registry
        self._save_registry()
        
        return model_id
    
    def get_all_models(self) -> List[Dict]:
        """Get all registered models, sorted by training date (newest first)."""
        models = self._registry.get("models", [])
        # Sort by training_date descending
        return sorted(models, key=lambda x: x.get("training_date", ""), reverse=True)
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by ID."""
        for model in self._registry.get("models", []):
            if model.get("id") == model_id:
                return model
        return None
    
    def get_model_by_path(self, model_path: str) -> Optional[Dict]:
        """Get a model by file path."""
        for model in self._registry.get("models", []):
            if model.get("file_path") == model_path:
                return model
        return None
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        models = self._registry.get("models", [])
        original_count = len(models)
        self._registry["models"] = [m for m in models if m.get("id") != model_id]
        
        if len(self._registry["models"]) < original_count:
            self._save_registry()
            return True
        return False
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a model's metadata.
        
        Args:
            model_id: Model ID to update
            updates: Dictionary of fields to update
        
        Returns:
            True if updated, False if not found
        """
        for model in self._registry.get("models", []):
            if model.get("id") == model_id:
                model.update(updates)
                self._save_registry()
                return True
        return False
    
    def search_models(
        self,
        name_filter: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        min_roc_auc: Optional[float] = None,
        feature_set: Optional[str] = None,
        horizon: Optional[int] = None
    ) -> List[Dict]:
        """
        Search models by various criteria.
        
        Args:
            name_filter: Filter by name (substring match)
            min_accuracy: Minimum accuracy threshold
            min_roc_auc: Minimum ROC AUC threshold
            feature_set: Filter by feature set name
            horizon: Filter by horizon (trading days)
        
        Returns:
            List of matching models
        """
        results = []
        
        for model in self._registry.get("models", []):
            # Name filter
            if name_filter and name_filter.lower() not in model.get("name", "").lower():
                continue
            
            # Accuracy filter
            test_metrics = model.get("metrics", {}).get("test", {})
            if min_accuracy is not None:
                accuracy = test_metrics.get("accuracy", 0.0)
                if accuracy < min_accuracy:
                    continue
            
            # ROC AUC filter
            if min_roc_auc is not None:
                roc_auc = test_metrics.get("roc_auc", 0.0)
                if roc_auc < min_roc_auc:
                    continue
            
            # Feature set filter
            if feature_set is not None:
                params = model.get("parameters", {})
                if params.get("feature_set") != feature_set:
                    continue
            
            # Horizon filter
            if horizon is not None:
                params = model.get("parameters", {})
                if params.get("horizon") != horizon:
                    continue
            
            results.append(model)
        
        return results

