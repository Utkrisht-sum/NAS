import os
import pandas as pd
from typing import Dict, Any, Tuple
from PIL import Image

def analyze_dataset(file_path_or_url: str) -> Dict[str, Any]:
    """
    Analyzes a dataset to automatically determine its type and properties.
    Returns: Dict containing 'type' (tabular, image, text), 'input_dim', 'num_classes', etc.
    """
    metadata = {
        "source": file_path_or_url,
        "type": "unknown",
        "input_dim": None,
        "num_classes": None,
        "dataset_size": None
    }

    # Simple check for predefined torchvision datasets
    predefined = ["mnist", "cifar10", "fashionmnist"]
    if isinstance(file_path_or_url, str) and file_path_or_url.lower() in predefined:
        metadata["type"] = "image"
        if file_path_or_url.lower() == "mnist":
            metadata["input_dim"] = (1, 28, 28)
            metadata["num_classes"] = 10
            metadata["dataset_size"] = 60000
        elif file_path_or_url.lower() == "cifar10":
            metadata["input_dim"] = (3, 32, 32)
            metadata["num_classes"] = 10
            metadata["dataset_size"] = 50000
        elif file_path_or_url.lower() == "fashionmnist":
            metadata["input_dim"] = (1, 28, 28)
            metadata["num_classes"] = 10
            metadata["dataset_size"] = 60000
        return metadata

    # Check for CSV/Tabular
    if isinstance(file_path_or_url, str) and file_path_or_url.endswith('.csv'):
        metadata["type"] = "tabular"
        try:
            df = pd.read_csv(file_path_or_url)
            # Assume last column is target for simplicity
            metadata["input_dim"] = len(df.columns) - 1
            target_col = df.columns[-1]
            metadata["num_classes"] = df[target_col].nunique()
            metadata["dataset_size"] = len(df)

            # Regression vs Classification heuristic
            if metadata["num_classes"] > 20 and pd.api.types.is_numeric_dtype(df[target_col]):
                metadata["task_type"] = "regression"
            else:
                metadata["task_type"] = "classification"
        except Exception as e:
            print(f"Error analyzing CSV: {e}")

    # Check for Image Directory
    elif isinstance(file_path_or_url, str) and os.path.isdir(file_path_or_url):
        metadata["type"] = "image"
        metadata["task_type"] = "classification"
        try:
            classes = [d for d in os.listdir(file_path_or_url) if os.path.isdir(os.path.join(file_path_or_url, d))]
            metadata["num_classes"] = len(classes)

            # Find a sample image to get dims
            sample_img_found = False
            total_size = 0
            for cls in classes:
                cls_dir = os.path.join(file_path_or_url, cls)
                images = os.listdir(cls_dir)
                total_size += len(images)

                if not sample_img_found and len(images) > 0:
                    img_path = os.path.join(cls_dir, images[0])
                    try:
                        with Image.open(img_path) as img:
                            channels = len(img.getbands())
                            width, height = img.size
                            metadata["input_dim"] = (channels, height, width)
                        sample_img_found = True
                    except Exception:
                        pass

            metadata["dataset_size"] = total_size
        except Exception as e:
            print(f"Error analyzing image directory: {e}")

    return metadata
