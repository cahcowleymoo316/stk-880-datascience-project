import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import logging

sys.path.append('src')
sys.path.append('src/visualization')




def main(logging):
    # load data
    logging.info("######## Loading data")
    dataset = pd.read_csv('data/raw/pima-indians-diabetes.csv')
    logging.info(("Number of rows in data -> {}".format(len(dataset)," rows")))
    # split the data into x and y
    X = dataset.iloc[:,0:8]
    y = dataset.iloc[:,8]

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2020)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename = "stk-cookiecutter-project.log")
    main(logging)
