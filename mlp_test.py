from activation_function import Tanh, Linear, ReLU, Sigmoid
from layer import Layer
from mlp import MLP
import numpy as np
from metrics import Metrics
import csv
from sklearn import preprocessing

# xor tests
inputs = []
oracles = []

filename = "Iris.csv"
n_oracle = 3

print(inputs[0])
#print(oracles)

def test_execute():
    mlp = MLP()
    metrics = Metrics()
    act_func1 = ReLU()
    act_func2 = Sigmoid()
    lr = 0.001
    f = open("logs.txt", "w")
    f.close()
    mlp.add(Layer(9, act_func=act_func1, n_inputs=len(inputs[0]), learning_rate=lr))
    #mlp.add(Layer(9, act_func=act_func1, n_inputs=8, learning_rate=lr))
    mlp.add(Layer(len(oracles[0]), act_func=act_func2, n_inputs=9, learning_rate=lr))
    #mlp.summary()
    
    mlp.fit(inputs, oracles, epochs=1000)
    for i in range(len(inputs)):
        out = mlp.run(inputs[i])
        print(f'case {i}: {np.round(out), oracles[i]}')
    #mlp.summary()
test_execute()