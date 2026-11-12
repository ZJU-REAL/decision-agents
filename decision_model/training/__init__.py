"""Training adapters; importing this package does not load a training runtime."""
from __future__ import annotations

from decision_model.training.bridge import BatchRecords, TrainingBridge, build_training_bridge

__all__ = ["BatchRecords", "TrainingBridge", "build_training_bridge"]
