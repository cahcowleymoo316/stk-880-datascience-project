import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import sys
import logging

sys.path.append('src')
sys.path.append('src/visualization')

from visualization.visualize import *

def compile_model(n_features):
    model=Sequential()
    # Dense layer with 12 nodes
    model.add(Dense(12, input_dim = n_features, activation = 'sigmoid'))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.3))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    # compile the model
    model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model




def fit_model(model, features, labels, n_epochs=10, n_batch = 10, val_split = 0.1):
    history = model.fit(features, labels, epochs = n_epochs, batch_size = n_batch, validation_split=val_split)
    return history


def main(logging):
    # create untrained model
    logging.info("####### Compiling Model")
    model = compile_model(8)
    # Load data
    logging.info("####### Loading Data")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    # train your model on the data
    logging.info("####### Model Fit")
    history = fit_model(model, X_train, y_train, n_epochs=50, n_batch=30, val_split=0.2)
    # get loss plot
    loss_plot(history)
    # save model as a pickle file
    model_path = 'models/stk_model_v1.h5'
    logging.info("Saving trained model in {}".format(model_path))
    history.model.save(model_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename = "stk-cookiecutter-project.log")
    main(logging)

