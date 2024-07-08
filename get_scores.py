
def get_regression_error(preds, Y):
    n = len(preds)
    return sum([(pred-y) for pred, y in zip(preds, Y)])/n


def get_classification_accurancy(preds, Y):
    correct = sum([1 for pred, y in zip(preds, Y) if pred==y])
    return correct / len(preds)
