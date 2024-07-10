from src.SVM.SvmModel import PrimalSVM, SVM
from get_scores import get_classification_accurancy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


iris = load_breast_cancer()
X, y = iris.data, iris.target
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVM(r=1, d=2, sigma=2, regularization_param=1, kernel="polynomial")
model.fit(X_train, y_train)
preds = model.predict(X_test)
accurancy = get_classification_accurancy(preds, y_test)
print("SVM with kernel accurancy: " + str(accurancy))

model = PrimalSVM(1, 300, 0.1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accurancy = get_classification_accurancy(preds, y_test)
print("Linear SVM accurancy: " + str(accurancy))
