import numpy as np
from layer import Layer
from losses import Loss
from metrics import Metrics
from tqdm import trange
import matplotlib.pyplot as plt

class MLP:

    layers  = []
    loss    = None
    eta     = 0.01
    errors  = []

    def __init__(self, layers):
        if layers != None:
            self.layers = layers

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, learning_rate: float, loss: Loss, metrics = []):
        self.loss = loss
        self.eta = learning_rate
        self.metrics = Metrics(metrics)
    
    def __forward(self, input):
        nets_for_layer = []
        outputs_for_layer = []
        inputs_for_layer = []
        input_for_layer = np.array(input)
        for k in range(len(self.layers)):
            net, output = self.layers[k].run(input_for_layer)
            nets_for_layer.append(net)
            outputs_for_layer.append(output)
            inputs_for_layer.append(input_for_layer)
            input_for_layer = output
        return nets_for_layer, outputs_for_layer, inputs_for_layer

    def __backward(self, error, nets, outputs, inputs):
        for k in range(len(self.layers) - 1, -1, -1):
            propagate_errors = self.layers[k].train(error, nets[k], outputs[k], inputs[k])
            error = propagate_errors

    def run(self, input: []):
        _, outputs, _ = self.__forward(input)
        return outputs[-1]

    def fit(self, input=[[]], oracle=[[]], epochs=0):
        bar = trange(epochs, desc='ML')
        self.errors = []
        for _ in bar:
            self.metrics.reset()
            global_error = 0
            for j in range(len(input)):
                nets, outputs, inputs = self.__forward(input[j])
                error = self.eta * self.loss.partial_derivative(np.array(oracle[j]), outputs[-1])
                global_error += self.loss.error(np.array(oracle[j]), outputs[-1])
                self.metrics.compute__results(round(oracle[j][0]), round(outputs[-1][0]))
                self.__backward(error, nets, outputs, inputs)
            self.errors.append(round(global_error/len(input), 2))
            bar.set_description(f'ML (loss={round(global_error/len(input), 2)}) (accuracy={round(self.metrics.accuracy(), 2)})')
            if round(global_error/len(input), 2) == 0.00:
                break

    def plot_error(self):
        plt.plot(self.errors)
        plt.show()

    def get_metrics(self):
        return self.metrics.get_best_accuracy()

    def summary(self):
        s = ""
        for layer in self.layers:
            s += f"{15*'-'} Layer {15*'-'}\n"
            s += layer.summary()
            s += "\n"
        return s
