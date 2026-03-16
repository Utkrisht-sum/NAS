import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from tqdm import tqdm
import time

class ProxyTrainer:
    def __init__(self, accelerator=None, device='cpu'):
        self.accelerator = accelerator
        self.device = device

    def train_model(self, model, train_loader, val_loader, epochs=1, subset_ratio=0.1):
        """
        Proxy training (1-3 epochs on subset) using Accelerate for efficiency.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Use Subset for faster proxy training
        if subset_ratio < 1.0:
            dataset_size = len(train_loader.dataset)
            subset_indices = torch.randperm(dataset_size)[:int(dataset_size * subset_ratio)]
            train_subset = Subset(train_loader.dataset, subset_indices)

            # Simple dataloader recreation to preserve batch size, shuffle, etc.
            train_loader = DataLoader(
                train_subset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers
            )

        if self.accelerator:
            model, optimizer, train_loader, val_loader = self.accelerator.prepare(
                model, optimizer, train_loader, val_loader
            )
        else:
            model.to(self.device)

        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if not self.accelerator:
                    data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                if not self.accelerator:
                    data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

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
                output = model(data)
                loss = criterion(output, target)

                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                running_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    if not self.accelerator:
                        data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {accuracy:.2f}%")
            if accuracy > best_acc:
                best_acc = accuracy
                # Save best model logic can go here

        return best_acc
