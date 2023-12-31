import numpy as np
from layer import Layer
import progressbar
from losses import MeanSquaredError
class MLP:
    layers  = []
    loss    = None
    eta     = 0.01
    def __init__(self, layers):
        if layers != None:
            self.layers = layers

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, learning_rate: float, loss: MeanSquaredError):
        self.loss = loss
        self.eta = learning_rate

    def run(self, input: np.ndarray):
        input_for_layer = np.array(input)
        for i in range(len(self.layers)):
            _, output = self.layers[i].run(input_for_layer)
            input_for_layer = output
        return output
    
    def fit(self, input=[[]], oracle=[[]], epochs=0, batch_size=0):
        bar = progressbar.ProgressBar()
        f = open("logs.txt", "w")
        f.close()
        for _ in bar(range(epochs)):
            f = open("logs.txt", "a")
            global_error = 0
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
                error = self.eta * self.loss.partial_derivative(np.array(oracle[j]), output)
                global_error += self.loss.error(np.array(oracle[j]), output)
                for k in range(len(self.layers) - 1, -1, -1):
                    propagate_errors = self.layers[k].train(error, nets_for_layer[k], outputs_for_layer[k], inputs_for_layer[k])
                    error = propagate_errors
            f.write(f"{global_error}\n")
            f.close()
                 
    def summary(self):
        for layer in self.layers:
            print(f"{10*'-'} Layer {10*'-'}")
            layer.summary()
            