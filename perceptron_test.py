from perceptron import Perceptron, ActivationFunction
import numpy as np
from metrics import Metrics
input = [
    np.array([4, 6]),
    np.array([6, 4]),
    np.array([2, 2]),
    np.array([0, 4]),
    np.array([2, 0]),
    np.array([0, 0]),
    np.array([6, 2]),
    np.array([4, 4]),
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
    act_func = ActivationFunction()
    p = Perceptron(2, act_func, learning_rate=0.01)
    for i in range(2000):
        for i in range(0, len(input)):
            net, output = p.run(input[i])
            p.train(oracle[i], net, output)

    for i in range(0, len(input)):
        _, out = p.run(input[i])
        print(out, oracle[i])
        metrics.compute_results(out, oracle[i])
    print(f"Accuracy: {metrics.accuracy()}")

        

test_execute()