import pandas as pd
import numpy as np
import os

class DatasetAnalyzer:
    @staticmethod
    def detect_category(data_source):
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                return "Tabular"
            elif os.path.isdir(data_source):
                # Check for images (very basic)
                has_images = any(f.endswith(('.png', '.jpg', '.jpeg')) for r, d, f in os.walk(data_source) for f in f)
                if has_images:
                    return "Image"
            elif data_source.endswith('.json') or data_source.endswith('.txt'):
                return "Text"
            # Time-series could be detected via schema/columns
        return "Unknown"

    @staticmethod
    def get_metadata(data_source, category):
        meta = {"source": data_source, "type": category}
        if category == "Tabular" and isinstance(data_source, str) and data_source.endswith('.csv'):
            try:
                df = pd.read_csv(data_source)
                meta["num_samples"] = len(df)
                meta["num_features"] = len(df.columns) - 1 # assuming 1 target
                # Try to guess task based on target column type
                target_col = df.columns[-1]
                if df[target_col].dtype in ['object', 'category'] or len(df[target_col].unique()) < 20:
                    meta["task"] = "classification"
                    meta["num_classes"] = len(df[target_col].unique())
                else:
                    meta["task"] = "regression"
                    meta["num_classes"] = 1
            except Exception as e:
                print(f"Error getting tabular metadata: {e}")
        elif category == "Image" and isinstance(data_source, str) and os.path.isdir(data_source):
             # Simple subfolder counting for classes
             try:
                 classes = [d for d in os.listdir(data_source) if os.path.isdir(os.path.join(data_source, d))]
                 meta["num_classes"] = len(classes)
                 meta["task"] = "image classification"
             except Exception as e:
                 print(f"Error getting image metadata: {e}")

        return meta
