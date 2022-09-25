
import numpy as np
import pickle
import argparse
import os


def model_testing():
    #loading the test data
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    #Separate the X_test from y_test.
        X_test, y_test = test_data

    with open('lr_model.pkl', 'rb') as f:
        log_reg_model = pickle.load(f)

    with open('svm_model.pkl', 'rb') as f:
        svm_clas_model = pickle.load(f)

    with open('knn_model.pkl', 'rb') as f:
        knn_clas_model = pickle.load(f)

    y_pred_lr = log_reg_model.predict(X_test)
    y_pred_svm = svm_clas_model.predict(X_test)
    y_pred_knn = knn_clas_model.predict(X_test)

    with open('y_pred_lr.pkl', 'wb') as f:
        pickle.dump(y_pred_lr, f)

    with open('y_pred_svm.pkl', 'wb') as f:
        pickle.dump(y_pred_svm, f)

    with open('y_pred_knn.pkl', 'wb') as f:
        pickle.dump(y_pred_knn, f)



if __name__ == '__main__':
    model_testing()