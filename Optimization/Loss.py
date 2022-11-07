import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.pre = None

    def forward(self, prediction_tensor, label_tensor):
        self.pre = prediction_tensor + np.finfo(np.float64).eps
        return np.sum(np.where(label_tensor == 1, - np.log(self.pre), 0))

    def backward(self, label_tensor):
        return - (np.divide(label_tensor, self.pre))
