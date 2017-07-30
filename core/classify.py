from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from core.data import VectorColumns, get_dataset_as_dataframe, vectorize_dataset, vectorize_from_string
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

class MLClassifier:
    def __init__(self, datafile):
        self.datafile = datafile
        self.pipeline = Pipeline([
            ('poly', PolynomialFeatures(2)),
            ('scaler', StandardScaler()),
            ('svm', SVC()),
        ])
        self.best_estimator = None
        self.best_score = 0.0

        self.params = {
            'svm__C': (0.1, 0.5, 1., 5., 10.),
            'svm__kernel' : ('linear', 'rbf', 'sigmoid')
        }

    def train_model(self):
        print("Training Email Address Classifier using Grid Search and SVMs...")
        X, Y = vectorize_dataset(get_dataset_as_dataframe(self.datafile))
        grid_search = GridSearchCV(self.pipeline, self.params, n_jobs=2, verbose=1)
        grid_search.fit(X, Y)

        self.best_estimator = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
        print("Done.")

    def predict_spam(self, email_address):
        X = vectorize_from_string(email_address)

        if self.best_estimator:
           return self.best_estimator.predict(X)
        else:
            print("MLClassifier.predict_spam: Model is not trained yet. Returning -1.")
            return -1.0