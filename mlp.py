import numpy as np
from layer import Layer
class MLP:
    layers: []
    
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def run(self, input: np.ndarray):
        input_for_layer = np.array(input)
        for i in range(len(self.layers)):
            _, output = self.layers[i].run(input_for_layer)
            input_for_layer = output
        return output
    
    def fit(self, input=[], oracle=[], epochs=0):

        for i in range(epochs):
            for j in range(len(input)):
                nets_for_layer = []
                outputs_for_layer = []
                inputs_for_layer = []
                input_for_layer = np.array(input[j])
                for k in range(len(self.layers)):
                    net, output = self.layers[k].run(input_for_layer)
                    nets_for_layer.append(net)
                    outputs_for_layer.append(output)
                    inputs_for_layer.append(input_for_layer)
                    input_for_layer = output
                error = np.array(oracle[j]) - output
                for k in range(len(self.layers) - 1, 0, -1):
                    propagate_errors = self.layers[k].train(error, nets_for_layer[k], outputs_for_layer[k], inputs_for_layer[k])
                    error = propagate_errors