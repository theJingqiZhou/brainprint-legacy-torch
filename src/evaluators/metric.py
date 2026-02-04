from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from src.utils.registry import register_module


@register_module(parent="evaluators")
def evaluator(config):
    return Evaluators(config)


class Evaluators:
    def __init__(self, config):
        pass

    def accuracy(self, target, output):
        return accuracy_score(target, output)

    def precision(self, target, output, average=None):
        return precision_score(target, output, average=average)

    def recall(self, target, output, average=None):
        return recall_score(target, output, average=average)

    def fscore(self, target, output, average=None):
        return f1_score(target, output, average=average)
