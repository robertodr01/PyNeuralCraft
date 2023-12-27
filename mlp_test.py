from activation_function import Tanh, Linear, ReLU, Sigmoid
from layer import Layer
from mlp import MLP
import numpy as np
from metrics import Metrics

# xor tests
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
oracle = [[0], [1], [1], [0]]

def test_execute():
    mlp = MLP()
    metrics = Metrics()
    act_func1 = ReLU()
    act_func2 = Sigmoid()
    lr = 0.1
    f = open("logs.txt", "w")
    f.close()
    mlp.add(Layer(5, act_func=act_func1, n_inputs=len(input[0]), learning_rate=lr))
    mlp.add(Layer(len(oracle[0]), act_func=act_func2, n_inputs=5, learning_rate=lr))
    #mlp.summary()
    
    mlp.fit(input, oracle, epochs=1000)
    for i in range(len(input)):
        out = mlp.run(input[i])
        print(f'case {i}: {np.round(out, 2), oracle[i]}')
    #mlp.summary()
test_execute()