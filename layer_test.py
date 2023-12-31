from activation_function import Sigmoid, ReLU
from layer import Layer
import numpy as np
from metrics import Metrics
from tqdm import tqdm
from sklearn import preprocessing
import csv

input = [
#     [4, 6],
#     [6, 4],
#     [2, 2],
#     [0, 4],
#     [2, 0],
#     [0, 0],
#     [6, 2],
#     [4, 4],
    [1,2],
    [2,3],
    [3,1],
    [4,4],
    [5,2],
    [2,5],
]
oracle = [
#     [1],
#     [1],
#     [0],
#     [0],
#     [0],
#     [0],
#     [1],
#     [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [1],
]

def test_execute():
    metrics = Metrics()
    act_func1 = ReLU()
    act_func2 = Sigmoid()
    lr = 0.01
    layer1 = Layer(4, act_func=act_func1, n_inputs=len(input[0]))
    layer2 = Layer(len(oracle[0]), act_func=act_func2, n_inputs=4)
    #layer3 = Layer(len(oracle[0]), act_func=act_func2, n_inputs=8)
    
    print("Layer 1:")
    layer1.summary()
    print("Layer 1:")
    layer2.summary()
    #print("Layer 1:")
    #layer3.summary()

    print()
    for i in tqdm(range(1)):
        for j in range(len(input[:10])):
            net1, output1 = layer1.run(np.array(input[j]))
            net2, output2 = layer2.run(output1)
            #net3, output3 = layer3.run(output2)
            errors = np.array(oracle[j]) - output2
            #propagate_errors = layer3.train(errors, net3, output3)
            propagate_errors = layer2.train(errors, net2, output2, output1)
            propagate_errors = layer1.train(propagate_errors, net1, output1, np.array(input[j]))
    print("Layer 1:")
    layer1.summary()
    print("Layer 1:")
    layer2.summary()
    #print("Layer 1:")
    #layer3.summary()
    print()

    for i in range(0, len(input[:10])):
        net1, output1 = layer1.run(np.array(input[i]))
        net2, output2 = layer2.run(output1)
        #net3, output3 = layer3.run(output2)
        print(f'case {i}: {output2, oracle[i]}') 

test_execute()