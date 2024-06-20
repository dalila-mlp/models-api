import ast
import argparse
import importlib
import os
import sys
import uuid

from typing import List, Tuple

import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.model_trainer import ModelTrainer_Tf, ModelTrainer_Sk, ModelTrainer_other


def determine_model_type_from_imports(file_path: str) -> str:
    with open(file_path, "r") as source:
        tree = ast.parse(source.read(), filename=file_path)
    
    imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
    import_froms = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None]

    return (
        "Sk" if any("sklearn" in module for module in imports + import_froms)
        else "Tf"
        if any("tensorflow" in module or "keras" in module for module in imports + import_froms) 
        else "Other"
    )

def convert_numpy_to_list(metrics: dict) -> dict:
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
    return metrics

def plot_and_log_metrics(metrics: dict) -> List[str]:
    plot_ids = []

    for metric, values in metrics.items():
        plt.figure(figsize=(10, 6))

        if isinstance(values, list) or isinstance(values, np.ndarray):
            epochs = range(len(values))
            plt.plot(epochs, values, label=f"{metric} over epochs")
        else:
            plt.plot([0, 1], [values, values], label=f"{metric} (constant)")

        plt.xlabel("Epoch" if isinstance(values, list) else "Index")
        plt.ylabel(metric)
        plt.title(f"{metric.capitalize()} Metric")
        plt.legend()
        plt.grid(True)
        plot_id = str(uuid.uuid4())
        plot_filename = f"{plot_id}.png"
        plt.savefig(plot_filename)
        plt.close()
        plot_ids.append(plot_id)
    
    return plot_ids

def load_model_class(temp_script_path: str) -> object:
    spec = importlib.util.spec_from_file_location("model", temp_script_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    for name, cls in model_module.__dict__.items():
        if isinstance(cls, type):
            return cls

def main(temp_script_path: str, test_size: float) -> Tuple[List[str], dict]:
    data = load_iris()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    y_train_tf = to_categorical(y_train, 3)
    y_test_tf = to_categorical(y_test, 3)
    model_class = load_model_class(temp_script_path)
    model_instance = model_class()
    model_type = determine_model_type_from_imports(temp_script_path)

    if model_type == "Tf":
        trainer = ModelTrainer_Tf(model_instance)
        y_train = y_train_tf
        y_test = y_test_tf
    elif model_type == "Sk":
        trainer = ModelTrainer_Sk(model_instance)
    else:
        trainer = ModelTrainer_other(model_instance)

    trainer.train(x_train, y_train)
    metrics = trainer.evaluate(x_test, y_test)
    metrics = convert_numpy_to_list(metrics)
    plot_ids = plot_and_log_metrics(metrics)

    return plot_ids, metrics
