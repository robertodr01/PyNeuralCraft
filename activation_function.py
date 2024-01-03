import numpy as np

class ActivationFunction:

    def output(self, x: np.ndarray) ->  np.ndarray:
        pass

    def derivative(self, x: np.ndarray) ->  np.ndarray:
        pass


class Linear(ActivationFunction):
    
    def output(self, x: np.ndarray):
        return x

    def derivative(self, x: np.ndarray):
        return np.full(x.size, 1)


class Sigmoid(ActivationFunction):
    
    def output(self, x: np.ndarray, slope:float=1):
        return 1 / (1 + np.exp(-(slope * x)))

    def derivative(self, x: np.ndarray):
        fx = self.output(x)
        return fx * (1 - fx)
    
class ReLU(ActivationFunction):

    def output(self, x: np.ndarray):
        return np.maximum(x, 0)
    
    def derivative(self, x: np.ndarray):
        return np.greater(x, 0)
    
class Tanh(ActivationFunction):

    def output(self, x: np.ndarray):
        return np.tanh(x)

    def derivative(self, x: np.ndarray):
        return np.subtract(1, np.power(np.tanh(x), 2))
    
def instantiate_act_func(act: str):
    if act == 'linear':
        return Linear()
    elif act == 'sigmoid':
        return Sigmoid()
    elif act == 'relu':
        return ReLU()
    elif act == 'tanh':
        return Tanh()
    else:
        raise Exception('no activation function found')
