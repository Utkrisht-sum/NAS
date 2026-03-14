import os
import urllib.request
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.config import Config
from utils.logger import logger

def load_dataset(dataset_source: str, batch_size: int = Config.DEFAULT_BATCH_SIZE) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Loads a dataset (predefined torchvision, CSV, or Image folder) and returns Train/Val DataLoaders and metadata.
    """

    # Predefined Datasets
    if isinstance(dataset_source, str) and dataset_source.lower() in ["mnist", "cifar10", "fashionmnist"]:
        return _load_predefined(dataset_source.lower(), batch_size)

    # CSV Datasets
    elif isinstance(dataset_source, str) and dataset_source.endswith(".csv"):
        return _load_csv(dataset_source, batch_size)

    # Image Folders
    elif isinstance(dataset_source, str) and os.path.isdir(dataset_source):
        return _load_image_folder(dataset_source, batch_size)

    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

def _load_predefined(name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, dict]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_dir = Config.PRETRAIN_DATASETS_DIR

    if name == "mnist":
        full_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        input_dim = (1, 28, 28)
        num_classes = 10
    elif name == "cifar10":
        transform_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_cifar)
        input_dim = (3, 32, 32)
        num_classes = 10
    elif name == "fashionmnist":
        full_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        input_dim = (1, 28, 28)
        num_classes = 10

    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "type": "image",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "dataset_size": len(full_dataset)
    }

    return train_loader, val_loader, metadata

def _load_csv(path: str, batch_size: int) -> Tuple[DataLoader, DataLoader, dict]:
    df = pd.read_csv(path)

    # Assume last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    num_classes = len(set(y))
    is_classification = num_classes < 20

    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y)
        y_tensor = torch.tensor(y, dtype=torch.long)
    else:
        y_tensor = torch.tensor(y, dtype=torch.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "type": "tabular",
        "input_dim": X.shape[1],
        "num_classes": num_classes if is_classification else 1,
        "dataset_size": len(df),
        "task_type": "classification" if is_classification else "regression"
    }

    return train_loader, val_loader, metadata

def _load_image_folder(path: str, batch_size: int) -> Tuple[DataLoader, DataLoader, dict]:
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Default resize for arbitrary folders to prevent OOM
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = datasets.ImageFolder(root=path, transform=transform)

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get shape from first item
    sample_img, _ = full_dataset[0]
    input_dim = tuple(sample_img.shape)

    metadata = {
        "type": "image",
        "input_dim": input_dim,
        "num_classes": len(full_dataset.classes),
        "dataset_size": len(full_dataset)
    }

    return train_loader, val_loader, metadata

def pretrain_download():
    """Download required datasets for pretraining if they don't exist."""
    logger.info("Initializing pre-training datasets...")
    try:
        _load_predefined("mnist", 1)
        logger.info("MNIST downloaded successfully.")

        # Download Iris dataset CSV
        iris_path = os.path.join(Config.PRETRAIN_DATASETS_DIR, "iris.csv")
        if not os.path.exists(iris_path):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            urllib.request.urlretrieve(url, iris_path)
            # Add headers
            df = pd.read_csv(iris_path, header=None)
            df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
            df.to_csv(iris_path, index=False)
            logger.info("Iris dataset downloaded successfully.")

    except Exception as e:
        logger.error(f"Failed to download pre-training datasets: {e}")
