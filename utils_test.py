from utils import k_fold_cross_validation, hold_out_validation
import numpy as np
def test_hold_out(X, y):
    d = hold_out_validation(X, y)
    # for p in d:
    #     print(p['X_train'])
    #     print(p['X_test'])
    #     print(p['y_train'])
    #     print(p['y_test'])

X = [
    [1,2,3,4,5,6],
    [1,2,3,4,5,6],
    [1,2,3,4,5,6],
    [1,2,3,4,5,6],
]
y = [
    [1,2],
    [1,2],
    [1,2],
    [1,2],
]
test_hold_out(np.array(X), np.array(y))