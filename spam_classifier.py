import argparse

import pickle

from core.classify import MLClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    action can either be train or predict.
    
    train: Used to train a spam email address classifier. Train takes an csv file similar to spam_test.csv as input (specified using the -f option) and stores the classifier model as a pickle.
    Example: 
    $ python3 spam_classifier.py train -f input.csv -mf output_model.bin
    $ python3 spam_classifier.py train -f input.csv 
    
    predict: Uses a pickled model to make a prediction. Requires the model binary file (-mf) and an email address (-e_. 
    Example:
    $ python3 spam_classifier.py predict -mf input_model.bin -e abc@gmail.com
    """)

    parser.add_argument("action")

    parser.add_argument("-f", "--file", help="The input CSV file. Must have the same format as spam_test.csv", type=str)
    parser.add_argument("-mf", "--modelfile", help="The binary for the classifier model.", type=str)
    parser.add_argument("-e", "--email", help="The email address used in the predict option.", type=str)

    args = parser.parse_args()

    if args.action == 'train':
        if not args.file:
            print("Incorrect usage of train. Needs an input csv file to train classifier (check usage using -h).")
        else:
            mlc = MLClassifier(args.file)
            mlc.train_model()
            pickle.dump(mlc, open(args.modelfile if args.modelfile else "model.bin", 'wb'))
    elif args.action == 'predict':
        if args.modelfile and args.email:
            mlc = pickle.load(open(args.modelfile, 'rb'))
            print(mlc.predict_spam(args.email))
