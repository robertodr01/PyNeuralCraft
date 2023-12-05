import activation_function
import numpy as np
import pytest

input = np.array([1, 2])

def test_Linear():
    function  = activation_function.Linear()
    assert np.array_equal(function.output(input), np.array([1, 2]))
    assert np.array_equal(function.derivative(input), np.array([1, 1]))


def test_Sigmoid():
    function  = activation_function.Sigmoid()
    x = np.around(function.output(input), 4)
    y = np.around(np.array([0.73105857863, 0.8807970779779]), 4)
    assert np.array_equal(x, y)

    # assert np.array_equal(function.derivative(input), np.array([0.73105857863, 0.8807970779779]))
