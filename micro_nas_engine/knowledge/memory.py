import json
import os
import time

class ArchitectureMemory:
    """
    Manages the persistent architecture memory.
    """
    def __init__(self, filepath="micro_nas_engine/knowledge/architecture_memory.json"):
        self.filepath = filepath
        self.memory = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.memory = json.load(f)
            except json.JSONDecodeError:
                self.memory = []
        else:
            self.memory = []

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def add_entry(self, dataset_meta, architecture_config, val_accuracy, compute_cost):
        entry = {
            "dataset_metadata": dataset_meta,
            "architecture_config": architecture_config,
            "validation_accuracy": val_accuracy,
            "compute_cost": compute_cost,
            "timestamp": time.time()
        }
        self.memory.append(entry)
        self.save()

    def get_all(self):
        return self.memory

    def get_for_dataset(self, dataset_type):
        return [entry for entry in self.memory if entry.get("dataset_metadata", {}).get("type") == dataset_type]
