import numpy as np
from activation_function import ActivationFunction
import random

random.seed(1)
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
        self.bias = random.random()
        for _ in range(n_inputs):
            value = random.random()
            self.weights = np.append(self.weights, value)

    def run(self, x: np.ndarray):
        if isinstance(x, np.ndarray) == False:
            raise ValueError("Output must be a numpy list")
        if x.size != self.weights.size:
            print(x.size, self.weights.size)
            raise ValueError("Number of inputs must be equal to number of weights")
        net: float = np.dot(x, self.weights) + self.bias
        out = self.act_func.output(np.array([net]))[0]
        return (net, out)
    
    def train(self, error: int, net: int, output: int) -> np.ndarray:
        print(error, net, output)
        weights = np.insert(self.weights, 0, self.bias)
        der = self.act_func.derivative(np.array([net]))[0] #* input
        print(der)
        print()
        delta = error * der * 1000
        weights = weights + self.eta * delta * output
        self.weights = weights[1:]
        self.bias = weights[0]
        #print([delta*1000]*self.weights.size)
        return (np.array([delta]*self.weights.size), self.weights)
    
    def summary(self):
        print(self.weights, self.bias)