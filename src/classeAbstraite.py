import json

class DynamicParams:
    def __init__(self):
        self.params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'max_leaves': 0,
            'min_samples_split': 3,
            'min_samples_leaf': 3,
            'hidden_units': 64,
            'num_hidden_layers': 2,
            'input_shape': 4,
            'optimizer': 'Adam',
            'loss': 'CategoricalCrossentropy',
            'n_neighbors': 20,
            'algorithm': 'auto',
            'leaf_size': 20,
            'trainable_base_layers': 10,
            'C': 1.0,
            'max_iter': 10,
            "solver": "liblinear",
            "n_estimators": 10,
            "kernel": "linear",
            "gamma": "auto",
            "conv_filters": 64,
            "kernel_size": 64,
            "pool_size": 32,
            "num_conv_layers": 2,
            'input_shape_CNN': (64, 64, 1)
        }

    def get_param(self, key):
        return self.params.get(key)

    def set_param(self, key, value):
        self.params[key] = value

    def save_params(self, filepath):
        with open(filepath, 'w') as file:
            json.dump(self.params, file)

    def load_params(self, filepath):
        with open(filepath, 'r') as file:
            self.params = json.load(file)