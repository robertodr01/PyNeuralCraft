from losses import instantiate_loss
from activation_function import instantiate_act_func
import json
from mlp import MLP
from layer import Layer
from itertools import product 
from copy import deepcopy
dest = "models"
class GridSearch:

    model_parameters    = []
    X_train             = []
    y_train             = []

    def read_json(self, json_files: []):
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)
                self.model_parameters.append(data)

    def set_dataset(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def test_grid(self):
        tests = []
        for model_parameter in self.model_parameters:
            model_tests = [{}]
            grid = model_parameter['grid_search']
            model_tests = create_combinations(model_tests, grid['lr'], 'learning_rate')
            model_tests = create_combinations(model_tests, grid['momentum'], 'momentum')
            model_tests = create_combinations(model_tests, grid['Nesterov'], 'Nesterov')
            model_tests = create_combinations(model_tests, grid['kernel_regularizer'], 'kernel_regularizer')
            model_tests = create_combinations(model_tests, grid['bias_regularizer'], 'bias_regularizer')
            for model_test in model_tests:
                model_test['layers']    = []
                model_test['name']      = model_parameter['name']
                model_test['epochs']    = model_parameter['epochs']
                model_test['loss']      = model_parameter['loss']
                model_test['metrics']   = model_parameter['metrics']
                for layer in model_parameter['layers']:
                    obj = {
                        'units'     : layer['units'],
                        'inputs'    : layer['inputs'],
                        'act_func'  : layer['activation'],
                    }
                    model_test['layers'].append(obj)
                tests.append(model_test)

        for i in range(len(tests)):
            print(tests[i])
            errors = self.run_model(tests[i])
            f = open(f"{dest}/{model_test['name']}-case:{i}-err:{str(round(min(errors), 2))}.logs", 'w')
            f.write(f"{str(tests[i])}\n")
            f.write(str(errors) + "\n")
            f.close()

    def run_model(self, test):
        layers = []
        for layer in test['layers']:
            layers.append(
                Layer(
                    layer['units'],
                    instantiate_act_func(layer['act_func']),
                    layer['inputs'],
                    test['kernel_regularizer'],
                    test['bias_regularizer'],
                    test['momentum'],
                    test['Nesterov']
                )
            )
        mlp = MLP(layers)
        mlp.compile(test['learning_rate'],instantiate_loss(test['loss']), test['metrics'])
        mlp.fit(self.X_train, self.y_train, test['epochs'])
        mlp.plot_error()
        return mlp.errors

def create_combinations(a, b, key):
    c = []
    for i in a:
        for j in b:
            obj = deepcopy(i)
            obj[key] = j
            c.append(obj)
    return c