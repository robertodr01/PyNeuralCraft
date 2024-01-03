import numpy as np

class Loss:
    def error(self, label: np.ndarray, output:np.ndarray):
        pass

    def partial_derivative(self, label: np.ndarray, output: np.ndarray):
        pass

class MeanSquaredError(Loss):
    def error(self, label: np.ndarray, output: np.ndarray):
        return np.mean(np.square(label - output))

    def partial_derivative(self, label, output):
        return label - output

class MeanEuclideanError(Loss):
    def error(self, label: np.ndarray, output:np.ndarray):
        return np.mean(np.linalg.norm(output - label))
    
    def partial_derivative(self, label: np.ndarray, output: np.ndarray):
        pass

def instantiate_loss(loss: str):
    if loss == 'mean_squared_error':
        return MeanSquaredError()
    else:
        raise Exception('no loss found')