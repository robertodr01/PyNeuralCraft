from perceptron import Linear
from layer import Layer
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
    metrics = Metrics()
    act_func = Linear()
    layer1 = Layer(3, act_func=act_func, n_inputs=len(input[0]))
    layer2 = Layer(5, act_func=act_func, n_inputs=3)
    layer3 = Layer(len(oracle[0]), act_func=act_func, n_inputs=5)
    for i in range(500):
        for i in range(0, len(input)):
            net1, output1 = layer1.run(np.array(input[i]))
            net2, output2 = layer2.run(output1)
            net3, output3 = layer3.run(output2)
            errors = np.array(oracle[i]) - output3
            propagate_errors = layer3.train(errors, net3, output3, output2)
            propagate_errors = layer2.train(propagate_errors, net2, output2, output1)
            propagate_errors = layer1.train(propagate_errors, net1, output1, np.array(input[i]))

    for i in range(0, len(input)):
        net1, output1 = layer1.run(np.array(input[i]))
        net2, output2 = layer2.run(output1)
        net3, output3 = layer3.run(output2)
        print(f'case {i}: {output3, oracle[i]}') 

test_execute()