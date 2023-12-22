import numpy as np
from perceptron import Perceptron
from activation_function import ActivationFunction

class Layer:
    perceptrons: []
    def __init__(self, n_perceptrons, act_func=None, n_inputs=None, learning_rate=0.001):
        if(n_perceptrons==None and n_inputs==None):
            raise ValueError("perceptrons or number of inputs must be greater than 0")
        if(n_inputs==None):
            raise ValueError("Number of inputs must be greater than 0")
        if(isinstance(act_func, ActivationFunction) == False):
            raise TypeError("Activation function must be an instance of ActivationFunction")
        self.perceptrons = []
        for _ in range(n_perceptrons):
            self.perceptrons.append(Perceptron(n_inputs, act_func, learning_rate))

    def run(self, x: np.ndarray) -> tuple:
        outs = np.array([])
        nets = np.array([])
        for i in range(len(self.perceptrons)):
            (net, out) = self.perceptrons[i].run(x)
            outs = np.append(outs, out)
            nets = np.append(nets, net)
        return (nets, outs)

    def train(self, errors: np.ndarray, net: np.ndarray, output: np.ndarray):
        propagate_errors = np.array([0.0] * self.perceptrons[0].weights.size)
        for i in range(len(self.perceptrons)):
            propagate_errors += self.perceptrons[i].train(errors[i], net[i], output[i])
            #print(propagate_errors)
        return propagate_errors
    
    def summary(self):
        for perceptron in self.perceptrons:
            print(f"{10*'-'} Perceptron {10*'-'}")
            perceptron.summary()
            print()