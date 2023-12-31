import numpy as np

def random_init(num_unit, num_input):
    input_weights = np.random.randn(num_input, num_unit) 
    bias_weights = np.random.randn(num_input, 1)
    return input_weights, bias_weights

def random_ranged_init(num_unit, num_input, range=(-0.7, 0.7), **kwargs):
    min_range, max_range = range[0], range[1]
    if min_range > max_range:
        raise ValueError('The min value must be <= than the max value')
    weights = np.random.uniform(low=min_range, high=max_range, size=(num_unit, num_input))
    bias = np.random.uniform(low=min_range, high=max_range, size=(num_input, 1))
    return weights, bias

def xavier_init(num_unit, num_input):
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(1 / num_input)
    return input_weights, bias_weights

def he_init(num_unit, num_input):
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(2 / num_input)
    return input_weights, bias_weights