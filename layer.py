import numpy as np
from perceptron import Perceptron, ActivationFunction

class Layer:
    perceptrons: []
    def __init__(self, n_perceptrons, act_func=None, n_inputs=None):
        if(n_perceptrons==None and n_inputs==None):
            raise ValueError("perceptrons or number of inputs must be greater than 0")
        if(n_inputs==None):
            raise ValueError("Number of inputs must be greater than 0")
        if(isinstance(act_func, ActivationFunction) == False):
            raise TypeError("Activation function must be an instance of ActivationFunction")
        self.perceptrons = []
        for _ in range(n_perceptrons):
            self.perceptrons.append(Perceptron(n_inputs, act_func))

    def run(self, x: np.ndarray) -> tuple:
        outs = np.array([])
        nets = np.array([])
        for p in self.perceptrons:
            (net, out) = p.run(x)
            outs = np.append(outs, out)
            nets = np.append(nets, net)
        return (nets, outs)

    def train(self, errors: np.ndarray, net: np.ndarray, output: np.ndarray, x: np.ndarray):
        propagate_errors = np.array([0] * self.perceptrons[0].weights.size)
        for i in range(len(self.perceptrons)):
            (delta, weights) = self.perceptrons[i].train(errors[i], net[i], output[i], x)
            for j in range(len(weights)):
                propagate_errors[j] += delta[j]*weights[j]
        return propagate_errors