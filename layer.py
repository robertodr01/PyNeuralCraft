import numpy as np
from activation_function import ActivationFunction

class Layer:

    def __init__(self, n_perceptrons, act_func=None, n_inputs=None, learning_rate=0.001):
        if(n_perceptrons==None and n_inputs==None):
            raise ValueError("perceptrons or number of inputs must be greater than 0")
        if(n_inputs==None):
            raise ValueError("Number of inputs must be greater than 0")
        if(isinstance(act_func, ActivationFunction) == False):
            raise TypeError("Activation function must be an instance of ActivationFunction")
        self.weights = np.random.rand(n_perceptrons, n_inputs)
        self.biases = np.random.rand(n_perceptrons, 1)
        self.act_func = act_func
        self.eta = learning_rate

    def run(self, x: np.ndarray) -> tuple:
        outs = np.array([])
        nets = np.array([])
        for i in range(len(self.weights)):
            net = np.dot(self.weights[i], x) + self.biases[i][0]
            out = self.act_func.output(np.array([net]))[0]
            outs = np.append(outs, out)
            nets = np.append(nets, net)
        return (nets, outs)

    def train(self, errors: np.ndarray, net: np.ndarray, output: np.ndarray, input):
        propagate_errors = np.array([0.0] * len(self.weights[0]))
        for i in range(len(self.weights)):
            der = self.act_func.derivative(np.array([net[i]]))[0]
            delta = errors[i] * der
            self.weights[i] += self.eta * delta * output[i] * input
            self.biases[i][0] += self.eta * delta * output[i]
            propagate_errors += delta * self.weights[i]
        return propagate_errors
    
    def summary(self):
        for i in range(len(self.weights)):
            print(f"{10*'-'} Perceptron {10*'-'}")
            print(self.weights[i])