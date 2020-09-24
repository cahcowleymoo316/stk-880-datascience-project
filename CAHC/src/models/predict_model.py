import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import sys

sys.path.append('src')
sys.path.append('src/visualization')

from visualization.visualize import *
import logging


def main(logging):
    # load data
    logging.info("######## Loading Data")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    logging.info("######## Loading Model")
    # load model
    model = load_model('models/stk_model_v1.h5')

    # check accuracy
    _, accuracy = model.evaluate(X_test, y_test, verbose = 0)
    logging.info("######## Evaluating Model")
    logging.info("Model Accuracy{}".format(accuracy))
    y_pred = model.predict_classes(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("######## Confusion Matrix >- {}".format(conf_matrix))
    logging.info("######## Plotting Confusion Matrix")
    # plot confusion_matrix
    plot_confusion_matrix(cm = conf_matrix, normalize = True, target_names = ['0', '1'], filepath = 'reports/figures/confusion_matrix.png')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="stk-cookiecutter-project.log")
    main(logging)