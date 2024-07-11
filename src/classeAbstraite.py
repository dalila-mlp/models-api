import json

class DynamicParams:
    def __init__(self):
        self.params = {
            # Common and Model-Specific Parameters
            'objective': 'multi:softmax',
            'epochs' : 100,
            'num_class': 8,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'max_leaves': 0,
            'min_samples_split': 3,
            'min_samples_leaf': 3,
            'hidden_units': 64,
            'num_hidden_layers': 5,
            'input_shape': 2,
            'optimizer': 'Adam',
            'loss': 'CategoricalCrossentropy',
            'n_neighbors': 20,
            'algorithm': 'auto',
            'leaf_size': 20,
            'trainable_base_layers': 10,
            'C': 1.0,
            'max_iter': 10,
            'solver': 'auto',
            'n_estimators': 10,
            'kernel': 'linear',
            'gamma': 'auto',
            'conv_filters': 64,
            'kernel_size': 64,
            'pool_size': 32,
            'num_conv_layers': 2,
            'input_shape_CNN': (64, 64, 1),
            'max_iter': 300,
            'tol': 1e-03,
            'alpha_1': 1e-06,
            'alpha_2': 1e-06,
            'lambda_1': 1e-06,
            'lambda_2': 1e-06,
            'compute_score': False,
            'fit_intercept': True,
            'normalize': False,
            'copy_X': True,
            'verbose': False,
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'degree': 2,
            'max_features': None,
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
