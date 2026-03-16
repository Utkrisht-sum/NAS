import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class TabularDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Assuming last column is target, others are features
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)

        target = df.iloc[:, -1]
        if target.dtype in [object, 'category', 'int64']:
            # Classification
            self.y = torch.tensor(target.astype('category').cat.codes.values, dtype=torch.long)
        else:
            # Regression
            self.y = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(dataset_path, dataset_type, batch_size=32):
    train_loader = None
    val_loader = None
    input_shape = None
    num_classes = None

    try:
        if dataset_type == "Tabular":
            dataset = TabularDataset(dataset_path)
            input_shape = dataset.X.shape[1]
            num_classes = len(torch.unique(dataset.y))

            # Simple split
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        elif dataset_type == "Image":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if os.path.exists(os.path.join(dataset_path, "train")):
                train_ds = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
                val_ds = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform)
            else:
                dataset = datasets.ImageFolder(dataset_path, transform=transform)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

            input_shape = (3, 224, 224)
            num_classes = len(train_ds.dataset.classes) if hasattr(train_ds, 'dataset') else len(train_ds.classes)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        elif dataset_type == "Text":
            # Very basic mock for text, real implementation would use HuggingFace Datasets
            pass

        elif dataset_type == "Time-series":
            # Very basic mock
            pass

    except Exception as e:
        print(f"Error loading dataset: {e}")

    return train_loader, val_loader, input_shape, num_classes
