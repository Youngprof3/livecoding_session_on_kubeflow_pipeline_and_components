
import numpy as np
import pickle
import argparse
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def data_modelling():
    #loading the train data
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    #Separate the X_train from y_train.
        X_train, y_train = train_data

        log_regres = LogisticRegression(max_iter=10000)
        lr_model = log_regres.fit(X_train, y_train)

        svm = SVC(kernel ="linear", random_state=2)
        svm_model = svm.fit(X_train, y_train)

        knn = KNeighborsClassifier(n_neighbors=7)
        knn_model = knn.fit(X_train, y_train)


    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)


if __name__ == '__main__':
    data_modelling() 
