from losses import instantiate_loss
from activation_function import instantiate_act_func
import json
from mlp import MLP
from layer import Layer
from itertools import product 
from copy import deepcopy

class ModelParameter:
    epochs          = 0
    loss            = None
    metrics         = []
    layers_params   = []
    grid_search     = None

    def __init__(self, model_param: dict):
        mlp_structure       = model_param['config']
        self.epochs         = mlp_structure['epochs']
        self.loss           = mlp_structure['loss']
        self.metrics        = mlp_structure['metrics']
        for layer in mlp_structure['layers']:
            self.layers_params.append(LayerParameter(layer))
        self.grid_search    = GridSearchParameter(model_param['grid_search'])


class GridSearchParameter:

    lr          = []
    momentum    = []
    Nesterov    = []
    kernel_reg  = []
    bias_reg    = []

    def __init__(self, grid_structure):
        self.lr         = grid_structure['lr']
        self.momentum   = grid_structure['momentum']
        self.Nesterov   = grid_structure['Nesterov']
        self.kernel_reg = grid_structure['kernel_regularizer']
        self.bias_reg   = grid_structure['bias_regularizer']


class LayerParameter:

    units       = 0
    inputs      = 0
    act_func    = 0

    def __init__(self, layer_structure: dict):
        self.units      = layer_structure['units']
        self.inputs     = layer_structure['inputs']
        self.act_func   = layer_structure['activation']

    def to_string(self):
        return f"units: {self.units}, inputs: {self.inputs}, act_func: {self.act_func}"


class GridSearch:

    file_name       = ''
    model_parameter = None
    X_train         = []
    y_train         = []

    def read_json(self, json_file):
        self.file_name = json_file
        with open(json_file, 'r') as file:
            data = json.load(file)
        self.model_parameter = ModelParameter(data)

    def set_dataset(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        
    def test_grid(self):
        tests = [{}]
        grid = self.model_parameter.grid_search

        tests = create_combinations(tests, grid.lr, 'learning_rate')
        tests = create_combinations(tests, grid.momentum, 'momentum')
        tests = create_combinations(tests, grid.Nesterov, 'Nesterov')
        tests = create_combinations(tests, grid.kernel_reg, 'kernel_reg')
        tests = create_combinations(tests, grid.bias_reg, 'bias_reg')

        for i in range(len(tests)):
            print(tests[i])
            errors = self.run_model(
                tests[i]['learning_rate'],
                tests[i]['momentum'],
                tests[i]['Nesterov'],
                tests[i]['kernel_reg'],
                tests[i]['bias_reg']
            )
            f = open(f"{self.file_name.split('.')[0]}-case:{i}-err:{str(round(min(errors), 2))}.logs", 'w')
            f.write(f"epochs: {str(self.model_parameter.epochs)}\n")
            for l in self.model_parameter.layers_params:
                f.write(f"{l.to_string()}\n")
            f.write(f"loss: {str(self.model_parameter.loss)}\n")
            f.write(f"{str(tests[i])}\n")
            f.write(str(errors) + "\n")
            f.close()

    def run_model(self, lr, momentum, Nesterov, kern_reg, bias_reg):
        layers = []
        mp = self.model_parameter
        for lp in mp.layers_params:
            layers.append(
                Layer(
                    lp.units,
                    instantiate_act_func(lp.act_func),
                    lp.inputs,
                    kern_reg,
                    bias_reg,
                    momentum,
                    Nesterov
                )
            )
        mlp = MLP(layers)
        mlp.compile(lr,instantiate_loss(mp.loss), mp.metrics)
        mlp.fit(self.X_train, self.y_train, mp.epochs)
        return mlp.errors

def create_combinations(a, b, key):
    c = []
    for i in a:
        for j in b:
            obj = deepcopy(i)
            obj[key] = j
            c.append(obj)
    return c