from core.data import VectorColumns
import numpy as np

class RuleBasedClassifier:
    def __init__(self):
        pass

    def evaluate(self, X):
        if len(X.shape) == 1:
            if (abs(X[VectorColumns.HAS_AT_SIGN] - 0) < 0.01) or (X[VectorColumns.PRE_AT_LENGTH] <= 2):
                return 1.0
            else:
                return 0.0
        else:
            Y = np.zeros(X.shape[0])
            for i, vector in enumerate(X):
                if (abs(vector[VectorColumns.HAS_AT_SIGN.value] - 0) < 0.01) or (vector[VectorColumns.PRE_AT_LENGTH.value] <= 2):
                    Y[i] = 1.0
                else:
                    Y[i] = 0.0
            return Y