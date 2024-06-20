import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
                             log_loss, jaccard_score, cohen_kappa_score, matthews_corrcoef)
from tensorflow.keras.losses import CategoricalCrossentropy

class ModelTrainer_Tf:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, x_train, y_train, epochs=10, batch_size=32, x_val=None, y_val=None):
        if x_val is not None and y_val is not None:
            self.classifier.train(x_train, y_train, validation_data=(x_val, y_val))
        else:
            self.classifier.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        y_pred = self.classifier.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_proba = np.eye(y_test.shape[1])[y_pred_classes]
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
        # Log charts
        self.log_charts(y_test, y_pred_proba)

class ModelTrainer_Sk:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, x_train, y_train):
        self.classifier.train(x_train, y_train)  # Changed from fit to train

    def evaluate(self, x_test, y_test):
        y_pred = self.classifier.predict(x_test)
        y_pred_proba = self.classifier.model.predict_proba(x_test) if hasattr(self.classifier.model, "predict_proba") else None  # Ensure to call predict_proba on the internal model

        y_test_classes = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
        y_pred_classes = y_pred if len(y_pred.shape) == 1 else np.argmax(y_pred, axis=1)

        metrics = {
            'accuracy': accuracy_score(y_test_classes, y_pred_classes),
            'precision': precision_score(y_test_classes, y_pred_classes, average='weighted'),
            'recall': recall_score(y_test_classes, y_pred_classes, average='weighted'),
            'f1_score': f1_score(y_test_classes, y_pred_classes, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_classes, y_pred_classes),
            'jaccard_index': jaccard_score(y_test_classes, y_pred_classes, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_test_classes, y_pred_classes),
            'matthews_corrcoef': matthews_corrcoef(y_test_classes, y_pred_classes)
        }

        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr'),
                'average_precision': average_precision_score(y_test, y_pred_proba, average='weighted'),
                'log_loss': log_loss(y_test, y_pred_proba),
            })

        return metrics

    def save_model(self, filename):
        import joblib
        joblib.dump(self.classifier, filename)

    def load_model(self, filename):
        import joblib
        self.classifier = joblib.load(filename)

class ModelTrainer_other:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, x_train, y_train, epochs=10, batch_size=32,  x_val=None, y_val=None):
        if x_val is not None and y_val is not None:
            self.classifier.train(x_train, y_train, validation_data=(x_val, y_val))
        else:
            self.classifier.train(x_train, y_train, epochs, batch_size)

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

    def save_model(self, filename):
        self.classifier.save(filename)

    def load_model(self, filename):
        self.classifier.load(filename)
