from perceptron import Perceptron
from activation_function import Sigmoid, ReLU
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
    # [1,2],
    # [2,3],
    # [3,1],
    # [4,4],
    # [5,2],
    # [2,5],
]
oracle = [
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1
    # 0,
    # 0,
    # 1,
    # 0,
    # 1,
    # 1,
]

def test_execute():
    metrics = Metrics()
    act_func = Sigmoid()
    p = Perceptron(len(input[0]), act_func, learning_rate=0.01)
    for i in range(1000):
        for i in range(0, len(input)):
            net, output = p.run(np.array(input[i]))
            error = oracle[i] - output
            p.train(error, net, output)

    for i in range(0, len(input)):
        _, out = p.run(np.array(input[i]))
        print(f'case {i}: {out, oracle[i]}')

test_execute()