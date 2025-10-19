from typing import Annotated, Tuple
from zenml import step

class Metrics:

    def __init__(self, TP, TN, FP, FN):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
        
    def accuracy(self):
        total = self.TP + self.TN + self.FP + self.FN
        print("Total : ", total)
        return (self.TP + self.TN) / total if total else 0.0
    
    def precision(self):
        denom = self.TP + self.FP
        return self.TP / denom if denom else 0.0
    
    def recall(self):
        denom = self.TP + self.FN
        return self.TP / denom if denom else 0.0
    
    def f1_score(self, precision = None, recall= None):
        print("Precision : ", precision)
        print("Recall : ", recall)
        if precision is None:
            precision = self.precision()
        if recall is None:
            recall = self.recall()
        
        denom = precision + recall
        return 2 * (precision * recall) / denom if denom else 0.0
    
@step
def confusion_matrix(TP : int, TN: int, FP: int , FN: int) -> Tuple[Annotated[float, "accuracy"],
                                                                Annotated[float, "precision"],
                                                                Annotated[float, "recall"],
                                                                Annotated[float, "f1_Score"]]:

    metrics = Metrics(TP, TN, FP, FN)
    accuracy = metrics.accuracy() 
    precision = metrics.precision()
    recall = metrics.recall()
    f1_score = metrics.f1_score(precision=precision, recall=recall)

    return accuracy, precision, recall, f1_score 
        