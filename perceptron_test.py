from perceptron import Perceptron, Linear
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
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1
]

def test_execute():
    metrics = Metrics()
    act_func = Linear()
    p = Perceptron(len(input[0]), act_func, learning_rate=0.001)
    for i in range(1000):
        for i in range(0, len(input)):
            net, output = p.run(np.array(input[i]))
            p.train(oracle[i], net, output, np.array(input[i]))

    for i in range(0, len(input)):
        _, out = p.run(np.array(input[i]))
        print(f'case {i}: {round(out), oracle[i]}')
        metrics.compute_results(round(out), oracle[i])
    print(f"Accuracy: {metrics.accuracy()}")    

test_execute()