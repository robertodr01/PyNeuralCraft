from activation_function import Tanh, Linear, ReLU, Sigmoid
from layer import Layer
from mlp import MLP
import numpy as np
from metrics import Metrics
import csv
from sklearn import preprocessing

inputs = []
oracles = []
#filename = "pima-indians-diabetes.data.csv"
filename = "simple_dataset.csv"
n_oracle = 1
with open(filename, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i=0
    for row in spamreader:
        i+=1
        row = [float(i) for i in row[0].split(",")]
        input=row[:len(row)-n_oracle]
        input = preprocessing.normalize([input])[0]
        inputs.append(input)
        oracle = row[-n_oracle:]
        oracles.append(oracle)
        if i == 100:
            break
print(inputs[1])
print(oracles[1])
input = [
    [4, 6],
    [6, 4],
    [2, 2],
    [0, 4],
    [2, 0],
    [0, 0],
    [6, 2],
    [4, 4],
]
oracle = [
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
]

def test_execute():
    mlp = MLP()
    metrics = Metrics()
    act_func1 = Sigmoid()
    act_func2 = Sigmoid()
    #out_act_func = Step()
    lr = 0.1
    f = open("logs.txt", "w")
    f.close()
    mlp.add(Layer(9, act_func=act_func1, n_inputs=len(input[0]), learning_rate=lr))
    #mlp.add(Layer(8, act_func=act_func1, n_inputs=9, learning_rate=lr))
    mlp.add(Layer(len(oracle[0]), act_func=act_func2, n_inputs=9, learning_rate=lr))
    mlp.summary()
    
    mlp.fit(input, oracle, epochs=20000)
    for i in range(len(input)):
        out = mlp.run(input[i])
        #print(inputs[i])
        print(f'case {i}: {np.round(out, 2), oracle[i]}')
    mlp.summary()
test_execute()