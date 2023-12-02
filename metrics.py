class Metrics:
    true_positive: int
    true_negative: int
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0


    def compute_results(self, out: float, oracle: float):
        if out == 1 and oracle == 1:
            self.true_pos += 1
        if out == 0 and oracle == 0:
            self.true_neg += 1    
        if out == 1 and oracle == 0:
            self.false_pos += 1
        if out == 0 and oracle == 1:
            self.false_neg += 1

    def accuracy(self):
        a = self.true_pos + self.true_neg 
        b = self.true_pos + self.true_neg + self.false_pos + self.false_neg
        return a / b