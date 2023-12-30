import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
class Metrics:
    
    true_positive: int
    true_negative: int
    accuracy_results = np.array([])
    metrics = []

    def __init__(self, metrics = []):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.confusion_matrix = pd.DataFrame()
        self.metrics = deepcopy(metrics)

    def reset(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.confusion_matrix = pd.DataFrame()

    def compute__results(self, out: float, oracle: float):
        if out == 1 and oracle == 1:
            self.true_pos += 1
        if out == 0 and oracle == 0:
            self.true_neg += 1    
        if out == 1 and oracle == 0:
            self.false_pos += 1
        if out == 0 and oracle == 1:
            self.false_neg += 1

    def compute_results(self, oracle: np.ndarray, out: np.ndarray):

        labels = np.unique(np.concatenate([oracle, out]))
        num_classes = len(labels)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred in zip(oracle, out):
            true_index = np.where(labels == true)[0][0]
            pred_index = np.where(labels == pred)[0][0]
            confusion_matrix[true_index, pred_index] += 1

        cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

        if num_classes == 2: 
            self.true_pos = cm.loc[1, 1]
            self.true_neg = cm.loc[0, 0]
            self.false_pos = cm.loc[0, 1]
            self.false_neg = cm.loc[1, 0]
        else:
            for label in labels:
                self.true_pos += cm.loc[label, label]
                self.true_neg += np.sum(np.delete(np.delete(cm.values, label, axis=0), label, axis=1))
                self.false_pos += np.sum(cm.loc[:, label]) - cm.loc[label, label]
                self.false_neg += np.sum(cm.loc[label, :]) - cm.loc[label, label]

        self.confusion_matrix = cm

    def accuracy(self):
        a = self.true_pos + self.true_neg 
        b = self.true_pos + self.true_neg + self.false_pos + self.false_neg
        res = 0 if b == 0 else a/b * 100
        self.accuracy_results = np.append(self.accuracy_results, res)
        return res
        
    def precision(self):
        a = self.true_pos
        b = self.true_pos + self.true_neg
        res = 0 if b == 0 else a/b
        return res
    
    def recall(self):
        a = self.true_pos
        b = self.true_pos + self.false_neg
        res = 0 if b == 0 else a/b
        return res
    
    def f1(self):
        a = self.precision() * self.recall()
        b = self.precision() + self.recall()
        res = 0 if b == 0 else 2 * (a / b)
        return res
    
    def plot(self):
        if 'accuracy' in self.metrics:
            plt.plot(self.accuracy_results)
            plt.show()