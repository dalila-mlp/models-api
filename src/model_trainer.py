import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models-draft')))
from src.classeAbstraite import DynamicParams
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, log_loss, jaccard_score, cohen_kappa_score, matthews_corrcoef
from tensorflow.keras.losses import CategoricalCrossentropy


class ModelTrainer_Tf:
    def __init__(self, classifier):
        self.classifier = classifier

    def validate_input_shape(self, x):
        expected_shape = self.classifier.model.input_shape
        if x.shape[1:] != expected_shape[1:]:
            raise ValueError(f"La dimension d'entrée {x.shape[1:]} n'est pas compatible avec la dimension attendue {expected_shape[1:]}")

    def train(self, x_train, y_train, epochs=10, batch_size=32,  x_val=None, y_val=None):
        self.validate_input_shape(x_train)
        if x_val is not None and y_val is not None:
            self.validate_input_shape(x_val)
            self.classifier.train(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                                  batch_size=batch_size)
        else:
            self.classifier.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test, num_classes):
        # Obtenez les prédictions du modèle
        y_pred = self.classifier.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_proba = np.eye(num_classes)[y_pred_classes]
        y_test_classes = np.argmax(y_test, axis=1)

        metrics = {
            'accuracy': accuracy_score(y_test_classes, y_pred_classes),
            'precision': precision_score(y_test_classes, y_pred_classes, average='weighted'),
            'recall': recall_score(y_test_classes, y_pred_classes, average='weighted'),
            'f1_score': f1_score(y_test_classes, y_pred_classes, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_classes, y_pred_classes),
            'roc_auc': roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'),
            'average_precision': average_precision_score(y_test, y_pred_proba, average='weighted'),
            'log_loss': log_loss(y_test, y_pred_proba),
            'categorical_crossentropy': CategoricalCrossentropy()(y_test, y_pred_proba).numpy(),
            'jaccard_index': jaccard_score(y_test_classes, y_pred_classes, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_test_classes, y_pred_classes),
            'matthews_corrcoef': matthews_corrcoef(y_test_classes, y_pred_classes)
        }

        '''# Calcul des courbes ROC et PR
        metrics['roc_curve'] = roc_curve(y_test.ravel(), y_pred_proba.ravel())
        metrics['precision_recall_curve'] = precision_recall_curve(y_test.ravel(), y_pred_proba.ravel())'''

        return metrics

    def save_model(self):
        self.classifier.save()

    def load_model(self, filename):
        self.classifier.load(filename)

class ModelTrainer_Sk:
    def __init__(self, classifier):
        self.classifier = classifier

    def validate_input_shape(self, x):
        if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'input_shape'):
            expected_shape = self.classifier.model.input_shape
            if x.shape[1:] != expected_shape[1:]:
                raise ValueError("La dimension d'entrée {x.shape[1:]} n'est pas compatible avec la dimension attendue {expected_shape[1:]}")
            else:
                print("Shape validation not applicable for non-Keras models")

    def train(self, x_train, y_train, x_val=None, y_val=None):
        self.validate_input_shape(x_train)

        y_train = np.argmax(y_train, axis=1)
        if y_val is not None:
            y_val = np.argmax(y_val, axis=1)

        if x_val is not None and y_val is not None:
            self.validate_input_shape(x_val)
            self.classifier.train(x_train, y_train, validation_data=(x_val, y_val))
        else:
            self.classifier.train(x_train, y_train)

    def evaluate(self, x_test, y_test, num_classes):
        y_pred = self.classifier.predict(x_test)

        # Handle the shape of y_pred_proba
        if y_pred.ndim == 1:
            y_pred_proba = np.zeros((y_pred.shape[0], num_classes))
            y_pred_proba[np.arange(y_pred.shape[0]), y_pred] = 1
        else:
            y_pred_proba = y_pred

        # Convert y_test to class indices if it's one-hot encoded
        y_test_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test_classes, y_pred),
            'precision': precision_score(y_test_classes, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_classes, y_pred, average='weighted'),
            'f1_score': f1_score(y_test_classes, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_classes, y_pred),
        }

        # Debug information
        print("Unique classes in y_test_classes:", np.unique(y_test_classes))
        print("Unique classes in y_pred:", np.unique(y_pred))

        unique_test_classes = np.unique(y_test_classes)
        unique_pred_classes = np.unique(y_pred)

        if len(unique_test_classes) > 1 and len(unique_pred_classes) > 1:
            try:
                # Calculate additional metrics only if more than one class is present
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
                metrics['average_precision'] = average_precision_score(y_test, y_pred_proba, average='weighted')
                metrics['log_loss'] = log_loss(y_test, y_pred_proba)
            except ValueError as e:
                print(f"Error calculating additional metrics: {e}")
                print("Skipping these metrics due to error.")
        else:
            print("ROC AUC and other metrics requiring multiple classes not calculated due to insufficient class presence in either y_test or y_pred.")

        return metrics


    def save_model(self):
        self.classifier.save()

    def load_model(self, filename):
        self.classifier.load(filename)

class ModelTrainer_other:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, x_train, y_train, x_val=None, y_val=None):
        if x_val is not None and y_val is not None:
            self.classifier.train(x_train, y_train, validation_data=(x_val, y_val))
        else:
            self.classifier.train(x_train, y_train)

    def evaluate(self, x_test, y_test):
        # Obtenez les prédictions du modèle
        y_pred = self.classifier.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_proba = np.eye(3)[y_pred_classes]
        y_test_classes = np.argmax(y_test, axis=1)

        metrics = {
            'accuracy': accuracy_score(y_test_classes, y_pred_classes),
            'precision': precision_score(y_test_classes, y_pred_classes, average='weighted'),
            'recall': recall_score(y_test_classes, y_pred_classes, average='weighted'),
            'f1_score': f1_score(y_test_classes, y_pred_classes, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_classes, y_pred_classes),
            'roc_auc': roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'),
            'average_precision': average_precision_score(y_test, y_pred_proba, average='weighted'),
            'log_loss': log_loss(y_test, y_pred_proba),
            'categorical_crossentropy': CategoricalCrossentropy()(y_test, y_pred_proba).numpy(),
            'jaccard_index': jaccard_score(y_test_classes, y_pred_classes, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_test_classes, y_pred_classes),
            'matthews_corrcoef': matthews_corrcoef(y_test_classes, y_pred_classes)
        }

        '''# Calcul des courbes ROC et PR
        metrics['roc_curve'] = roc_curve(y_test.ravel(), y_pred_proba.ravel())
        metrics['precision_recall_curve'] = precision_recall_curve(y_test.ravel(), y_pred_proba.ravel())'''

        return metrics

    def save_model(self):
        self.classifier.save()

    def load_model(self, filename):
        self.classifier.load(filename)