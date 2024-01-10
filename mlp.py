import numpy as np
from layer import Layer
from losses import Loss
from metrics import Metrics
from tqdm import trange
import pickle

class MLP:

    layers  = []
    loss    = None
    eta     = 0.01
    errors  = []

    def __init__(self, layers=None):
        if layers != None:
            self.layers = layers
            # for i in range(len(self.layers)):
            #     self.layers[i].n_processes = n_processes

    def add(self, layer: Layer):
        #layer.n_processes = self.n_processes
        self.layers.append(layer)

    def compile(self, learning_rate: float, loss: Loss, metrics = []):
        self.loss = loss
        self.eta = learning_rate
        self.metrics = metrics
    
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
        print(self.layers)
        print(outputs)
        return outputs[-1]

    def evaluate(self, inputs: [], oracles: [], adapt=None):
        error = 0
        metrics = Metrics(self.metrics)
        for input, oracle in zip(inputs, oracles):
            _, outputs, _ = self.__forward(input)
            if adapt != None:
                oracle = adapt([oracle])[0]
                outputs[-1] = adapt([outputs[-1]])[0]
            error += self.loss.error(np.array(oracle), outputs[-1])
            if len(self.metrics) > 0:
                    metrics.compute__results(oracle, outputs[-1])
        error = round(error/len(inputs), 6)
        if len(self.metrics) > 0:
            return error, metrics.accuracy()
        return error

    def fit(self, input=[[]], oracle=[[]], epochs=0, adapt=None):
        #bar = trange(epochs, desc='ML')
        errors = []
        accuracy = []
        metrics = Metrics(self.metrics)
        for _ in range(epochs):
            global_error = 0
            for j in range(len(input)):
                nets, outputs, inputs = self.__forward(input[j])
                if adapt != None:
                    oracle[j] = adapt([oracle[j]])[0]
                    outputs[-1] = adapt([outputs[-1]])[0]
                loss_partial_derivative = self.loss.partial_derivative(np.array(oracle[j]), outputs[-1])
                #loss_partial_derivative = clip_delta(loss_partial_derivative)
                error = self.eta * loss_partial_derivative
                self.__backward(error, nets, outputs, inputs)
                global_error += self.loss.error(np.array(oracle[j]), outputs[-1])
                if len(self.metrics) > 0:
                    metrics.compute__results(oracle[j], outputs[-1])
            global_error = round(global_error/len(input), 6)
            errors.append(global_error)
            accuracy.append(metrics.accuracy())
            #bar.set_description(f'ML (loss={round(global_error/len(input), 2)})')
        if len(self.metrics) > 0:
            return error, accuracy
        return errors

    def summary(self):
        s = ""
        for layer in self.layers:
            s += f"{15*'-'} Layer {15*'-'}\n"
            s += layer.summary()
            s += "\n"
        return s
    
    def save(self, filename):
        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)
    
    def load(self, filename):
        file = open(filename, 'rb')
        mlp = pickle.load(file)
        self.layers = mlp.layers
        self.loss    = mlp.loss
        self.eta     = mlp.eta
        file.close()
        

def clip_delta(grad, clip_threshold=1e5):
    grad_norm = np.linalg.norm(grad, ord=2)
    if grad_norm >= clip_threshold:
        num = clip_threshold/grad_norm
        grad = num * grad
    return grad