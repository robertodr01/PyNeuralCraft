import numpy as np
from activation_function import ActivationFunction

class Layer:
    
    # core
    weights     = []
    biases      = []
    act_func    = None
    # regularization
    lambda_w    = 0
    lambda_b    = 0
    # momentum
    delta_w_old = []
    delta_b_old = []
    alpha       = 0
    #nesterov
    vel_w_old   = 0
    vel_b_old   = 0
    Nesterov    = False

    def __init__(
        self, 
        n_perceptrons,
        act_func=None,
        n_inputs=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        momentum=None,
        Nesterov=False
    ):
        if(n_perceptrons==None and n_inputs==None):
            raise ValueError("perceptrons or number of inputs must be greater than 0")
        if(n_inputs==None):
            raise ValueError("Number of inputs must be greater than 0")
        if(isinstance(act_func, ActivationFunction) == False):
            raise TypeError("Activation function must be an instance of ActivationFunction")
        self.weights = np.random.rand(n_perceptrons, n_inputs)
        self.biases = np.random.rand(n_perceptrons, 1)
        self.act_func = act_func
        if kernel_regularizer != None:
            self.lambda_w = kernel_regularizer
        if bias_regularizer != None:
            self.lambda_b = bias_regularizer
        if momentum != None:
            self.alpha = momentum
        self.Nesterov = Nesterov
        self.delta_w_old = np.zeros((n_perceptrons, n_inputs))
        self.delta_b_old = np.zeros((n_perceptrons, 1))
        

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
            delta_w = delta * output[i] * input
            delta_b = delta * output[i]
            reg_w = regularization(self.lambda_w, self.weights[i])
            reg_b = regularization(self.lambda_b, self.biases[i][0])
            mom_w = momentum(self.alpha, self.delta_w_old[i])
            mom_b = momentum(self.alpha, self.delta_b_old[i])

            vel_w = 1
            vel_b = 1
            if self.Nesterov:
                vel_w = nesterov(self.vel_w_old, mom_w, delta_w)
                self.vel_w_old = vel_w
                vel_b = nesterov(self.vel_b_old, mom_b, delta_b)
                self.vel__b_old = vel_b
        
            self.delta_w_old[i] = delta_w
            self.delta_b_old[i] = delta_b
            self.weights[i] += delta_w - reg_w + mom_w * vel_w
            self.biases[i][0] += delta_b - reg_b + mom_b * vel_b
            propagate_errors += delta * self.weights[i]
            
        return propagate_errors
    
    def summary(self):
        for i in range(len(self.weights)):
            print(f"{10*'-'} Perceptron {10*'-'}")
            print(self.weights[i])

# optimization
def regularization(lam, weights):
    return 2 * lam * weights

def momentum(alpha, delta_old):
    return alpha * delta_old

def nesterov(velocity_old, mom, delta):
    return velocity_old * mom + delta