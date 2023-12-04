import numpy as np
class ActivationFunction:

    def output(self, x: np.ndarray):
        pass

    def derivative(self, x: np.ndarray):
        pass

class Linear(ActivationFunction):

    def output(self, x: np.ndarray):
        return x

    def derivative(self, x: np.ndarray):
        return [1] * x.size
class Perceptron:

    act_func: ActivationFunction
    weights: np.ndarray
    bias: float
    n_inputs: int
    eta = int

    def __init__(self, n_inputs: int, act_func: ActivationFunction, learning_rate=0.001):
        if(isinstance(act_func, ActivationFunction) == False):
            raise TypeError("Activation function must be an instance of ActivationFunction")
        if(n_inputs < 1):
            raise ValueError("Number of inputs must be greater than 0")
        self.act_func = act_func
        self.weights = np.array([])
        self.n_inputs = n_inputs
        self.eta = learning_rate
        self.bias = np.random.uniform(low=0.1, high=0.5)
        for _ in range(n_inputs):
            value = np.random.uniform(low=0.1, high=0.5)
            self.weights = np.append(self.weights, value)

    def run(self, x: np.ndarray):
        if isinstance(x, np.ndarray) == False:
            raise ValueError("Output must be a numpy list")
        if x.size != self.weights.size:
            raise ValueError("Number of inputs must be equal to number of weights")
        net: float = np.dot(x, self.weights) + self.bias*1
        out = self.act_func.output(np.array([net]))[0]
        return (net, out)
    
    def train(self, error: int, net: int, output: int, input: np.ndarray) -> np.ndarray:
        input = np.insert(input, 0, 1)
        weights = np.insert(self.weights, 0, self.bias)
        der = self.act_func.derivative(np.array([net]))[0] * input
        delta = error * der
        weights = weights + self.eta * np.dot(delta, output)
        self.weights = weights[1:]
        self.bias = weights[0]
        return (delta, self.weights)