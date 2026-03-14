import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Callable

from utils.config import Config
from utils.logger import logger

class Trainer:
    def __init__(self, model: nn.Module, task_type: str = "classification",
                 learning_rate: float = Config.DEFAULT_LEARNING_RATE,
                 device: str = Config.DEVICE,
                 progress_callback: Callable[[int, int, float, float], None] = None):
        """
        Initializes the trainer.
        progress_callback: A function taking (current_epoch, total_epochs, train_loss, val_metric)
        """
        self.device = device
        self.model = model.to(self.device)
        self.task_type = task_type

        # Loss function
        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.progress_callback = progress_callback

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = Config.DEFAULT_EPOCHS) -> float:
        """
        Trains the model for the specified number of epochs.
        Returns the best validation metric (accuracy for classification, negative MSE for regression).
        """
        best_val_metric = -float('inf')

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            # Use tqdm if no GUI callback is provided, else silent
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if self.progress_callback is None else train_loader

            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                if self.task_type == "classification":
                    # For CrossEntropy, target should be LongTensor of class indices
                    if batch_y.dim() > 1 and batch_y.shape[1] == 1:
                        batch_y = batch_y.squeeze(1)
                    batch_y = batch_y.long()

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if isinstance(pbar, tqdm):
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = running_loss / len(train_loader)

            # Evaluate after epoch
            val_metric = self.evaluate(val_loader)

            if val_metric > best_val_metric:
                best_val_metric = val_metric

            if self.progress_callback is not None:
                self.progress_callback(epoch + 1, epochs, avg_train_loss, val_metric)
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Metric: {val_metric:.4f}")

        return best_val_metric

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluates the model on the validation set.
        Returns Accuracy (0-1) for classification, or -MSE for regression.
        """
        self.model.eval()
        total = 0
        correct_or_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)

                if self.task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    if batch_y.dim() > 1 and batch_y.shape[1] == 1:
                        batch_y = batch_y.squeeze(1)
                    total += batch_y.size(0)
                    correct_or_loss += (predicted == batch_y).sum().item()
                else:
                    loss = self.criterion(outputs, batch_y)
                    total += batch_y.size(0)
                    correct_or_loss += loss.item() * batch_y.size(0)

        if self.task_type == "classification":
            accuracy = correct_or_loss / total
            return float(accuracy)
        else:
            # Negative MSE so that higher is better (for maximization in NAS)
            mse = correct_or_loss / total
            return -float(mse)
