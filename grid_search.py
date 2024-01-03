import json
from copy import deepcopy

def create_test(json_files: []):
    model_parameters = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            model_parameters.append(data)
    tests = []
    for model_parameter in model_parameters:
        model_tests = [{}]
        grid = model_parameter['grid_search']
        model_tests = create_combinations(model_tests, grid['lr'], 'learning_rate')
        model_tests = create_combinations(model_tests, grid['momentum'], 'momentum')
        model_tests = create_combinations(model_tests, grid['Nesterov'], 'Nesterov')
        model_tests = create_combinations(model_tests, grid['kernel_regularizer'], 'kernel_regularizer')
        model_tests = create_combinations(model_tests, grid['bias_regularizer'], 'bias_regularizer')
        model_tests = create_combinations(model_tests, grid['weights_initializer'], 'weights_initializer')
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
    return tests

def create_combinations(a, b, key):
    c = []
    for i in a:
        for j in b:
            obj = deepcopy(i)
            obj[key] = j
            c.append(obj)
    return c