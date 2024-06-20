import ast
import importlib
import sys
import os

# Add the current directory and the 'models' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from matplotlib import pyplot as plt
import numpy as np
from model_trainer import ModelTrainer_Tf, ModelTrainer_Sk, ModelTrainer_other
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse
import uuid

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
        plt.figure(figsize=(10, 6))

        if isinstance(values, list) or isinstance(values, np.ndarray):
            epochs = range(len(values))
            plt.plot(epochs, values, label=f'{metric} over epochs')
        else:
            plt.plot([0, 1], [values, values], label=f'{metric} (constant)')

        plt.xlabel('Epoch' if isinstance(values, list) else 'Index')
        plt.ylabel(metric)
        plt.title(f'{metric.capitalize()} Metric')
        plt.legend()
        plt.grid(True)

        plot_id = str(uuid.uuid4())
        plot_filename = f"{plot_id}.png"
        plt.savefig(plot_filename)
        plt.close()

        plot_ids.append(plot_id)
    
    return plot_ids

def load_model_class(temp_script_path):
    spec = importlib.util.spec_from_file_location("model", temp_script_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    for name, cls in model_module.__dict__.items():
        if isinstance(cls, type):
            return cls

def main(test_size, temp_script_path):

    from sklearn.datasets import load_iris
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model and log results.')
    parser.add_argument('temp_script_path', type=str, help='Path to the temporary model script')
    parser.add_argument('test_size', type=float, help='Test size for train-test split')
    parser.add_argument('transaction_id', type=str, help='Transaction ID')
    args = parser.parse_args()
    main(args.temp_script_path, args.test_size, args.transaction_id)