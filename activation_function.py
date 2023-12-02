import numpy as np

class ActivationFunction:

    def output(self, x: np.ndarray):
        pass

    def derivative(self, x: np.ndarray):
        pass



class Linear(ActivationFunction):
    
    def output(self, x: np.ndarray):
        return x;

    def derivative(self, x: np.ndarray):
        return 1;


class Sigmoid(ActivationFunction):
    
    def output(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

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