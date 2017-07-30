import os
from unittest import TestCase
import pandas as pd
import sys

from core.data import get_dataset_as_dataframe, vectorize_dataset, has_at_sign, pre_at_length

sys.argv = ['', '--pytest', '-l']


class DataTest(TestCase):
    def setUp(self):
        print("Setting up DataTest...")
        self.datafile = "../test_dataset.csv"
        self.df = pd.read_csv(self.datafile, names=['email_address', 'is_spam'])

    def test_get_dataset_as_dataframe(self):
        df = get_dataset_as_dataframe(self.datafile)

        self.assertEqual(len(df), 9, "Length of df is {}. Should be 9.".format(len(df)))
        self.assertMultiLineEqual(df.at[0, 'email_address'], "b19@gmail.com", "First email address does not match.")

    def test_has_at_sign(self):
        positive_case = "e9b0f521a11ecd1ff560@gmail.com"
        negative_case = "gmail.com"

        self.assertTrue(has_at_sign(positive_case), "has_at_sign failed for string {}".format(positive_case))
        self.assertFalse(has_at_sign(negative_case), "has_at_sign failed for string {}".format(negative_case))

    def tear_pre_at_length(self):
        short = pre_at_length("f@gmail.com")
        self.assertEqual(short, 1, msg="pre_at_length failed at {}, Returned length: {}.".format("f@gmail.com", short))

        long = pre_at_length("7dfb68997ecdc9755f90gmail.com")
        self.assertEqual(long, 0, msg="pre_at_length failed at {}, Returned length: {}.".format("7dfb68997ecdc9755f90gmail.com", long))


    def test_vectorize_dataset(self):
        X, Y = vectorize_dataset(self.df)

        self.assertAlmostEqual(Y[0], 0.0, places=4)
        self.assertAlmostEqual(Y[1], 0.0, places=4)
        self.assertAlmostEqual(Y[2], 0., places=4)
        self.assertAlmostEqual(Y[3], 0., places=4)
        self.assertAlmostEqual(Y[4], 0., places=4)
        self.assertAlmostEqual(Y[5], 1., places=4)
        self.assertAlmostEqual(Y[6], 0., places=4)
        self.assertAlmostEqual(Y[7], 1., places=4)
