import json
import os
from datetime import datetime
from typing import Dict, List, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.logger import logger

class MemoryManager:
    def __init__(self):
        self.memory_file = Config.MEMORY_FILE
        self.memory: List[Dict[str, Any]] = self.load_memory()

    def load_memory(self) -> List[Dict[str, Any]]:
        """Loads architecture memory from disk."""
        if not os.path.exists(self.memory_file):
            logger.info("No existing architecture memory found. Creating empty memory.")
            return []
        try:
            with open(self.memory_file, "r") as f:
                memory = json.load(f)
            logger.info(f"Loaded {len(memory)} architectures from memory.")
            return memory
        except Exception as e:
            logger.error(f"Failed to load architecture memory: {e}")
            return []

    def save_memory(self):
        """Saves current architecture memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=4)
            logger.info(f"Saved {len(self.memory)} architectures to memory.")
        except Exception as e:
            logger.error(f"Failed to save architecture memory: {e}")

    def add_entry(self, dataset_meta: Dict[str, Any], best_arch: Dict[str, Any],
                  accuracy: float, compute_cost: float):
        """Adds a successful architecture to memory."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset_metadata": dataset_meta,
            "architecture": best_arch,
            "accuracy": accuracy,
            "compute_cost": compute_cost
        }
        self.memory.append(entry)
        self.save_memory()

    def get_past_architectures(self, dataset_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves past architectures matching the current dataset type."""
        matching = [
            m for m in self.memory
            if m.get("dataset_metadata", {}).get("type") == dataset_type
        ]

        # Sort by best accuracy
        matching.sort(key=lambda x: x.get("accuracy", 0.0), reverse=True)
        return [m["architecture"] for m in matching[:limit]]
