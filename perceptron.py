import numpy as np
class ActivationFunction:

    def output(self, x: np.ndarray):
        pass

    def derivative(self, x: np.ndarray):
        pass


class Perceptron:

    act_func: ActivationFunction
    weights: list
    n_inputs: int
    eta = int

    def __init__(self, n_inputs: int, act_func: ActivationFunction, learning_rate: float):
        if(isinstance(act_func, ActivationFunction) == False):
            raise TypeError("Activation function must be an instance of ActivationFunction")
        if(n_inputs < 1):
            raise ValueError("Number of inputs must be greater than 0")
        self.act_func = act_func
        self.weights = []
        self.n_inputs = n_inputs
        self.eta = learning_rate
        for _ in range(n_inputs):
            self.weights.append(np.random.uniform(low=0.1, high=0.5))

    def run(self, x: np.ndarray):
        if isinstance(x, np.ndarray) == False:
            raise ValueError("Output must be a numpy list")
        if len(x) != len(self.weights):
            raise ValueError("Number of inputs must be equal to number of weights")
        net: float = np.dot(x, self.weights)
        out = self.act_func.output(np.array([net]))[0]
        return (net, out)
    
    def train(self, oracle: int, net: int, output: int) -> np.ndarray:
        if isinstance(oracle, (int, float)) == False:
            raise ValueError("Oracle must be a int or float")
        if isinstance(output, (int, float)) == False:
            raise ValueError("Output must be a int or float")
        der = self.act_func.derivative(np.array([net]))[0]
        delta = (oracle - output) * der
        for i in range(len(self.weights)):
            self.weights[i] += self.eta * np.dot(delta, output)
        return delta