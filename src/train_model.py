import ast
import importlib
import sys
import os
import inspect

# Add the current directory and the 'models' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from matplotlib import pyplot as plt
import numpy as np
from src.model_trainer import ModelTrainer_Tf, ModelTrainer_Sk, ModelTrainer_other
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.classeAbstraite import DynamicParams
import polars as pl
import pyarrow
import uuid
from pathlib import Path

ap = Path(__file__).parent.parent.resolve()

"""def get_model_class(module_name, class_name):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_"""

def determine_model_type_from_imports(file_path):
    with open(file_path, "r") as source:
        tree = ast.parse(source.read(), filename=file_path)
    
    imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
    import_froms = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None]

    if any('sklearn' in module for module in imports + import_froms):
        return 'Sk'
    elif any('tensorflow' in module or 'keras' in module for module in imports + import_froms):
        return 'Tf'
    else:
        return 'Other'

def convert_numpy_to_list(metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
    return metrics

def plot_and_log_metrics(metrics):
    plot_ids = []
    for metric, values in metrics.items():
        if isinstance(values, np.ndarray) and values.ndim > 1:
            # For matrices or high-dimensional data, we skip or handle separately
            print(f"Skipping plotting for {metric} due to incompatible dimensions.")
            continue
        
        plt.figure(figsize=(10, 6))

        # Check if values are scalar or list/array-like
        if np.isscalar(values):
            # Plot scalar values as horizontal lines
            plt.plot([0, 1], [values, values], label=f'{metric} (constant)')
            plt.xlabel('Index')
        elif isinstance(values, (list, np.ndarray)):
            # Check dimensions and plot accordingly
            values = np.array(values)
            if values.ndim == 1:
                epochs = range(len(values))
                plt.plot(epochs, values, label=f'{metric} over epochs')
            elif values.ndim == 2:
                for i in range(values.shape[1]):
                    epochs = range(values.shape[0])
                    plt.plot(epochs, values[:, i], label=f'{metric} series {i}')
            else:
                print(f"Skipping {metric} due to unsupported number of dimensions: {values.ndim}")
                continue
        else:
            print(f"Unhandled data type for metric {metric}: {type(values)}")
            continue

        plt.ylabel(metric)
        plt.title(f'{metric.capitalize()} Metric')
        plt.legend()
        plt.grid(True)

        plot_id = str(uuid.uuid4())
        plot_filename = f"{plot_id}.png"
        plt.savefig(f"{ap}/charts/{plot_filename}")
        plt.close()

        plot_ids.append(plot_id)

    return plot_ids

def load_model_class(temp_script_path):
    spec = importlib.util.spec_from_file_location("model", temp_script_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Get the base filename without extension to match class name
    base_filename = os.path.splitext(os.path.basename(temp_script_path))[0]

    for name, cls in model_module.__dict__.items():
        # Check if item is a class and defined in the module
        if inspect.isclass(cls) and cls.__module__ == model_module.__name__:
            return cls
    return None

def main(temp_script_path, dataset_temp_path, target_column,features, test_size):

    # Step 1: Load the dataset with Polars
    df = pl.read_parquet(dataset_temp_path)
    
    # Assuming 'target' is your label column and it's the last column
    x = df[features]
    y = df.select(target_column).to_numpy().flatten()  # Converting to NumPy for compatibility with sklearn

    # Step 2: Split the data using sklearn (Polars can be used if the entire workflow stays in Polars)
    x_train, x_test, y_train, y_test = train_test_split(x.to_pandas(), y, test_size=test_size, random_state=42)
    
    y = y.astype(int)
    num_classes = np.max(y) + 1
    y_train_tf = to_categorical(y_train, num_classes=num_classes)
    y_test_tf = to_categorical(y_test, num_classes=num_classes)

    params = DynamicParams()

    model_class = load_model_class(temp_script_path)
    model_instance = model_class(params)

    model_type = determine_model_type_from_imports(temp_script_path)

    if model_type == "Tf":
        trainer = ModelTrainer_Tf(model_instance)
        y_train = y_train_tf
        y_test = y_test_tf
    elif model_type == "Sk":
        trainer = ModelTrainer_Sk(model_instance)
        y_train = y_train_tf
        y_test = y_test_tf
    else:
        trainer = ModelTrainer_other(model_instance)

    trainer.train(x_train, y_train)
    metrics = trainer.evaluate(x_test, y_test, num_classes)
    metrics = convert_numpy_to_list(metrics)

    plot_ids = plot_and_log_metrics(metrics)

    return plot_ids, metrics