import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models-draft')))
from src.classeAbstraite import DynamicParams
from src.models_draft.classification_models.Sk_DecisionTreeClassifier_Dalila import DecisionTree
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

        # Calcul des courbes ROC et PR
        metrics['roc_curve'] = roc_curve(y_test.ravel(), y_pred_proba.ravel())
        metrics['precision_recall_curve'] = precision_recall_curve(y_test.ravel(), y_pred_proba.ravel())

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

    def evaluate(self, x_test, y_test):
        # Obtenez les prédictions du modèle
        y_pred = self.classifier.predict(x_test)
        y_pred_classes = y_pred
        y_pred_proba = np.eye(3)[y_pred_classes]
        y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[1] > 1 else y_test

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

        # Calcul des courbes ROC et PR
        metrics['roc_curve'] = roc_curve(y_test.ravel(), y_pred_proba.ravel())
        metrics['precision_recall_curve'] = precision_recall_curve(y_test.ravel(), y_pred_proba.ravel())

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

        # Calcul des courbes ROC et PR
        metrics['roc_curve'] = roc_curve(y_test.ravel(), y_pred_proba.ravel())
        metrics['precision_recall_curve'] = precision_recall_curve(y_test.ravel(), y_pred_proba.ravel())

        return metrics

    def save_model(self):
        self.classifier.save()

    def load_model(self, filename):
        self.classifier.load(filename)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Charger les données Iris
    data = load_iris()
    x = data.data
    y = data.target

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    from tensorflow.keras.utils import to_categorical

    # # Convertir les étiquettes de classe en format one-hot encoding
    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)

    # Créer une instance du modèle XGBoostClassifier
    params = DynamicParams()
    # rendre automatique  --soa
    model = DecisionTree(params)

    type = model.getTypeModel()

    # Créer une instance de ModelTrainer
    if type == "Tf":
        trainer = ModelTrainer_Tf(model)
    elif type == "Sk":
        trainer = ModelTrainer_Sk(model)
    else:
        trainer = ModelTrainer_other(model)

    # Entraîner le modèle
    trainer.train(x_train, y_train)

    # Évaluer le modèle
    metrics = trainer.evaluate(x_test, y_test)
    print("\nMetrics:", metrics)

    # Sauvegarder le modèle
    trainer.save_model()