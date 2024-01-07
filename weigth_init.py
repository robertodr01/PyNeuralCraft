import numpy as np

def random_init(num_unit, num_input):
    input_weights = np.random.randn(num_unit, num_input) 
    bias_weights = np.random.randn(num_unit, 1)
    return input_weights, bias_weights

def random_ranged_init(num_unit, num_input, range=(-0.7, 0.7), **kwargs):
    min_range, max_range = range[0], range[1]
    if min_range > max_range:
        raise ValueError('The min value must be <= than the max value')
    weights = np.random.uniform(low=min_range, high=max_range, size=(num_unit, num_input))
    bias = np.random.uniform(low=min_range, high=max_range, size=(num_unit, 1))
    return weights, bias

def xavier_init(num_unit, num_input):
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(1 / num_input)
    return input_weights, bias_weights

def he_init(num_unit, num_input):
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(2 / num_input)
    return input_weights, bias_weights

def xavier_uniform_init(num_unit, num_input):
    a = np.sqrt(6.0 / (num_input + num_unit))
    input_weights = np.random.uniform(low=-a, high=a, size=(num_unit, num_input))
    bias_weights = np.zeros((num_unit, 1))
    return input_weights, bias_weights

def instantiate_initializer(initializer: str):
    if initializer == "he_init":
        return he_init
    elif initializer == "xavier_init":
        return xavier_init
    elif initializer == "random_ranged_init":
        return random_ranged_init
    elif initializer == "random_init":
        return random_init
    elif initializer == "xavier_uniform_init":
        return xavier_uniform_init
    else:
        raise Exception('no initialization found')