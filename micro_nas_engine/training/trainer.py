import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from tqdm import tqdm
import time
import sys

from micro_nas_engine.utils.hardware import get_device

class ProxyTrainer:
    def __init__(self, accelerator=None, device='cpu'):
        self.accelerator = accelerator
        self.device = device

    def _execute_with_memory_guard(self, model, train_loader, optimizer, criterion):
        """
        AirLLM-inspired chunked execution / gradient checkpointing approach
        for a single epoch, with dynamic batch scaling on OOM.
        """
        try:
            # Enable gradient checkpointing if model supports it (for CNNs and deep nets)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            for batch_idx, (data, target) in enumerate(train_loader):
                if not self.accelerator:
                    data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                # Chunked Forward/Backward Pass Simulation
                # Real chunking requires rewriting the forward pass, so we rely on
                # accelerate / amp to manage memory, and catch OOMs.
                output = model(data)
                loss = criterion(output, target)

                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()

                # Free memory immediately
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return True, train_loader

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("CUDA OOM detected. Attempting dynamic batch scaling...")

                # Halve the batch size
                new_batch_size = max(1, train_loader.batch_size // 2)
                if new_batch_size == train_loader.batch_size:
                    print("Batch size already 1. Forcing CPU fallback.")
                    self.device = get_device(fallback_to_cpu=True)
                    model.to(self.device)
                    # Create new dataloader with batch size 1 but on CPU
                    new_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=train_loader.num_workers
                    )
                    return False, new_loader

                print(f"Reduced batch size to {new_batch_size}")

                # Create a new dataloader with the smaller batch size
                new_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=new_batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers
                )

                if self.accelerator:
                    new_loader = self.accelerator.prepare(new_loader)

                return False, new_loader
            else:
                raise e

    def train_model(self, model, train_loader, val_loader, epochs=1, subset_ratio=0.1):
        """
        Proxy training (1-3 epochs on subset) using Accelerate for efficiency.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        if subset_ratio < 1.0:
            dataset_size = len(train_loader.dataset)
            subset_indices = torch.randperm(dataset_size)[:int(dataset_size * subset_ratio)]
            train_subset = Subset(train_loader.dataset, subset_indices)
            train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)

        if self.accelerator:
            model, optimizer, train_loader, val_loader = self.accelerator.prepare(
                model, optimizer, train_loader, val_loader
            )
        else:
            model.to(self.device)

        model.train()

        current_loader = train_loader
        for epoch in range(epochs):
            success = False
            attempts = 0
            while not success and attempts < 3:
                attempts += 1
                success, current_loader = self._execute_with_memory_guard(model, current_loader, optimizer, criterion)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                if not self.accelerator:
                    data, target = data.to(self.device), target.to(self.device)

                try:
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                except RuntimeError as e:
                     if "out of memory" in str(e):
                         torch.cuda.empty_cache()
                         # Skip batch on OOM during validation to prevent crash
                         continue

        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy

class FullTrainer:
    def __init__(self, accelerator=None, device='cpu'):
        self.accelerator = accelerator
        self.device = device

    def train_model(self, model, train_loader, val_loader, epochs=10):
        """
        Full dataset training for the best architecture.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if self.accelerator:
            model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )
        else:
            model.to(self.device)

        best_acc = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                if not self.accelerator:
                    data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                try:
                    output = model(data)
                    loss = criterion(output, target)

                    if self.accelerator:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()

                    optimizer.step()
                    running_loss += loss.item()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print("OOM during full training, skipping batch to survive.")
                        continue

            scheduler.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    if not self.accelerator:
                        data, target = data.to(self.device), target.to(self.device)
                    try:
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                             torch.cuda.empty_cache()
                             continue

            accuracy = 100 * correct / total if total > 0 else 0
            print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {accuracy:.2f}%")
            if accuracy > best_acc:
                best_acc = accuracy

        return best_acc
