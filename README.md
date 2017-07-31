
```
usage: 
    $ spam_classifier.py [-h] [-f FILE] [-mf MODELFILE] [-e EMAIL] action

    
    action can either be train or predict.
    
    train: Used to train a spam email address classifier. Train takes an csv file similar to spam_test.csv as input (specified using the -f option) and stores the classifier model as a pickle.
    Example: 
    $ python3 spam_classifier.py train -f input.csv -mf output_model.bin
    $ python3 spam_classifier.py train -f input.csv 
    
    predict: Uses a pickled model to make a prediction. Requires the model binary file (-mf) and an email address (-e_. 
    Example:
    $ python3 spam_classifier.py predict -mf input_model.bin -e abc@gmail.com
    

    positional arguments: action
    
    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  The input CSV file. Must have the same format as
                            spam_test.csv
      -mf MODELFILE, --modelfile MODELFILE
                            The binary for the classifier model.
      -e EMAIL, --email EMAIL
                            The email address used in the predict option.

