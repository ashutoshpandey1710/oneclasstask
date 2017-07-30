import sys
from unittest import TestCase
import pandas as pd
from sklearn.metrics import accuracy_score

from core.classify import RuleBasedClassifier
from core.data import vectorize_dataset

sys.argv = ['', '--pytest', '-l']


class RuleBasedClassifierTest(TestCase):
    def setUp(self):
        print("Setting up RuleBasedClassifierTest...")
        self.datafile = "../spam_test.csv"
        self.df = pd.read_csv(self.datafile, names=['email_address', 'is_spam'])

    def test_rule_based_calssifier(self):
        X, Y = vectorize_dataset(self.df)

        classifier = RuleBasedClassifier()
        Y_pred = classifier.evaluate(X)

        accuracy = accuracy_score(Y, Y_pred)
        print("Accuracy Score: {}".format(accuracy))
        self.assertGreaterEqual(accuracy, 0.8, "Accuracy score is below threshold. Accuracy Score: {}.".format(accuracy))