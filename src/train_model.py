import ast
import importlib
import sys
import os
import inspect
import pickle

from sklearn.metrics import ConfusionMatrixDisplay

# Add the current directory and the 'models' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from src.model_trainer import ModelTrainer_Tf, ModelTrainer_Sk, ModelTrainer_other
from sklearn.model_selection import learning_curve, train_test_split
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

def plot_metrics(history):
    plot_ids = []
    metrics = ['accuracy', 'loss']
    plt.figure(figsize=(10, 6))

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()

        plot_id = str(uuid.uuid4())
        plot_filename = f"{plot_id}.png"
        plt.savefig(f"{ap}/charts/{plot_filename}")
        plt.close()

        plot_ids.append(plot_id)

    return plot_ids

def plot_confusion_matrix(confusion_matrix, class_names):
    plot_ids = []

    confusion_matrix = np.array(confusion_matrix)  # Ensure it's a NumPy array
    
    # Determine the indices of the existing classes in the confusion matrix
    existing_class_indices = np.where(confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) > 0)[0]
    existing_class_names = [class_names[i] for i in existing_class_indices]
    
    # Filter the confusion matrix to include only the existing classes
    filtered_confusion_matrix = confusion_matrix[np.ix_(existing_class_indices, existing_class_indices)]

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=filtered_confusion_matrix, display_labels=existing_class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    
    plot_id = str(uuid.uuid4())
    plot_filename = f"{plot_id}.png"
    plt.savefig(f"{ap}/charts/{plot_filename}")
    plt.close()
    plot_ids.append(plot_id)
    return plot_ids

def plot_regression(metrics, model,X_test, y_test):
    plot_ids = []

    # 1. Plot Regression Metrics as horizontal bars
    plt.figure(figsize=(10, 6))
    metric_names = list(metrics.keys())
    values = list(metrics.values())
    plt.barh(metric_names, values, color='skyblue')
    plt.xlabel('Metric Values')
    plt.title('Regression Metrics Overview')
    plot_id = str(uuid.uuid4())
    plot_filename = f"{plot_id}.png"
    plt.savefig(f"{ap}/charts/{plot_filename}")
    plt.close()
    plot_ids.append(plot_id)

    # 2. Predicted vs. Actual Plot
    predictions = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Predicted vs Actual Values')
    plt.legend()
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

def main(type, temp_script_path, dataset_temp_path, target_column, features, test_size, model_id, request_model_type, parameters):
    if type == 'train':

        # Step 1: Load the dataset with Polars
        df = pl.read_csv(dataset_temp_path)
        
        # Assuming 'target' is your label column and it's the last column
        x = df[features]
        y = df.select(target_column).to_numpy().flatten()  # Converting to NumPy for compatibility with sklearn

        # Determine unique class names from y, if categorical
        if request_model_type == "Classification":
            class_names = np.unique(y).astype(str)  # Convert class labels to string if necessary

        # Step 2: Split the data using sklearn (Polars can be used if the entire workflow stays in Polars)
        x_train, x_test, y_train, y_test = train_test_split(x.to_pandas(), y, test_size=test_size, random_state=42)
        y = y.astype(int)
        num_classes = np.max(y) + 1
        y_train_encoded = to_categorical(y_train, num_classes=num_classes)
        y_test_encoded = to_categorical(y_test, num_classes=num_classes)

        params = DynamicParams()
        for param_name, param_value in parameters.items():
            params.set_param(param_name, param_value)
        params.set_param("num_class", num_classes)
        params.set_param("input_shape", x.shape[1])

        model_class = load_model_class(temp_script_path)
        model_instance = model_class(params)
        response_model_type = determine_model_type_from_imports(temp_script_path)

        if response_model_type == "Tf":
            trainer = ModelTrainer_Tf(model_instance)
            history = trainer.train(x_train, y_train_encoded)
            trainer.save_model(model_id)
            metrics = trainer.evaluate(x_test, y_test_encoded, num_classes)
            metrics = convert_numpy_to_list(metrics)
            plot_ids = plot_metrics(history)
        elif response_model_type == "Sk":
            if request_model_type == "Classification":
                trainer = ModelTrainer_Sk(model_instance)
                history = trainer.train_classification(x_train, y_train_encoded)
                trainer.save_model(model_id)
                metrics = trainer.evaluate_classification(x_test, y_test_encoded, num_classes)
                metrics = convert_numpy_to_list(metrics)
                plot_ids = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
            elif request_model_type == "Regression":
                trainer = ModelTrainer_Sk(model_instance)
                history = trainer.train_regression(x_train, y_train)
                trainer.save_model(model_id)
                metrics = trainer.evaluate_regression(x_test, y_test)
                metrics = convert_numpy_to_list(metrics)
                plot_ids = plot_regression(metrics,model_instance,x_test,y_test)
            else :
                raise Exception('Model type not defined')
        else:
            trainer = ModelTrainer_other(model_instance)
            history = trainer.train(x_train, y_train)
            trainer.save_model(model_id)
            metrics = trainer.evaluate_classification(x_test, y_test, num_classes)
            metrics = convert_numpy_to_list(metrics)
            plot_ids = plot_confusion_matrix(metrics['confusion_matrix'], class_names)

        return plot_ids, metrics, response_model_type
    else:

        # Step 1: Load the dataset with Polars
        df = pl.read_csv(dataset_temp_path)
        
        # Assuming 'target' is your label column and it's the last column
        x = df[features]
        x = x.to_pandas()

        if "keras" in temp_script_path:
            model = tf.keras.models.load_model(temp_script_path)
            result = model.predict(x)
            result = np.argmax(result, axis=1)
        else:
            with open(temp_script_path, 'rb') as file:
                model = pickle.load(file)
                result = model.predict(x)

            # Convertir x_train en DataFrame polars
        if isinstance(x, pl.DataFrame):
            x_df = x
        else:
            x_df = pl.DataFrame(x)

        # Convertir les résultats en DataFrame polars
        result_df = pl.DataFrame({'prediction': result})

        # Combiner x_train avec les prédictions
        combined_df = x_df.hstack(result_df)

        # Convertir le DataFrame polars en une liste de dictionnaires
        result_json_serializable = combined_df.to_dicts()

        return result_json_serializable
