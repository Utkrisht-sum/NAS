import torch
import torch.nn as nn
from typing import Dict, Any

from metrics.compute_cost import estimate_compute_cost, estimate_flops
from models.model_builder import build_model
from utils.logger import logger
from utils.config import Config

def evaluate_architecture(arch: Dict[str, Any], input_dim: Any, num_classes: int,
                          train_loader, val_loader, task_type: str = "classification",
                          epochs: int = Config.DEFAULT_EPOCHS,
                          progress_callback=None) -> Dict[str, Any]:
    """
    Given an architecture definition, builds the model, estimates compute cost,
    trains it briefly, and evaluates accuracy.
    """
    logger.info(f"Evaluating architecture: {arch}")

    try:
        model = build_model(arch, input_dim, num_classes)

        # Calculate Compute Cost
        compute_cost = estimate_compute_cost(model)
        flops = estimate_flops(model)

        # Train and Evaluate
        from training.trainer import Trainer
        trainer = Trainer(model, task_type=task_type, progress_callback=progress_callback)
        best_accuracy = trainer.train(train_loader, val_loader, epochs=epochs)

        return {
            "architecture": arch,
            "accuracy": best_accuracy,
            "compute_cost": compute_cost,
            "flops": flops
        }

    except Exception as e:
        logger.error(f"Failed to evaluate architecture {arch}: {e}")
        # Return a poor score if evaluation fails (e.g., OOM, bad shape)
        return {
            "architecture": arch,
            "accuracy": -1.0,
            "compute_cost": float('inf'),
            "flops": float('inf')
        }
