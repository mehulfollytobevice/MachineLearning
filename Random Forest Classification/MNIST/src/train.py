# train.py
import argparse
import os

import joblib
import pandas as pd
import numpy as np
from sklearn import metrics

import config
import model_dispatcher


def run(fold, model):
    """
    This function takes in a fold number, builds the model and evaluates it's performance
    :param fold: fold data
    :param model: model to be used
    """

    # read training data
    df=pd.read_csv(config.TRAINING_FILE)

    # training data 
    df_train=df[df.kfold!=fold].reset_index(drop=True)

    # validation data
    df_valid=df[df.kfold==fold].reset_index(drop=True)

    # convert to numpy arrays
    x_train=df_train.drop('target',axis=1).values
    y_train=df_train.target.values

    x_valid=df_valid.drop('target',axis=1).values
    y_valid=df_valid.target.values

    # fetch the model from the model dispatcher
    clf=model_dispatcher.models[model]

    # fit the model on the training data
    clf.fit(x_train,y_train)

    # create predictions
    preds=clf.predict(x_valid)

    # calculate and print accuracy
    accuracy=metrics.accuracy_score(y_valid,preds)
    print(f"Fold={fold}, Accuracy={accuracy}, Model Name={model}")

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL,f"dt_{fold}_{model}.bin")
    )

if __name__=="__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args=parser.parse_args()

    run(fold=args.fold, model=args.model)

