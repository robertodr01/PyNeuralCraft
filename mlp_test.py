from activation_function import Linear
from layer import Layer
from mlp import MLP
import numpy as np
from metrics import Metrics
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
    [1, 1],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 1],
    [1, 1],
]

def test_execute():
    mlp = MLP()
    metrics = Metrics()
    act_func = Linear()
    mlp.add(Layer(3, act_func=act_func, n_inputs=len(input[0])))
    mlp.add(Layer(5, act_func=act_func, n_inputs=3))
    mlp.add(Layer(len(oracle[0]), act_func=act_func, n_inputs=5))

    mlp.fit(input, oracle, epochs=5000)

    for i in range(0, len(input)):
        out = mlp.run(input[i])
        print(f'case {i}: {out, oracle[i]}')

test_execute()