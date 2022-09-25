
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix

def model_evaluation():
    #loading the test data
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    #Separate the X_test from y_test.
        X_test, y_test = test_data

    with open('y_pred_lr.pkl', 'rb') as f:
        y_pred_lr = pickle.load(f)
    with open('y_pred_svm.pkl', 'rb') as f:
        y_pred_svm = pickle.load(f)
    with open('y_pred_knn.pkl', 'rb') as f:
        y_pred_knn = pickle.load(f)


    class_report_lr = classification_report(y_test, y_pred_lr)
    class_report_svm = classification_report(y_test, y_pred_svm)
    class_report_knn = classification_report(y_test, y_pred_knn)


    print(f'Classification report for the logistic regression model: {class_report_lr}')

    print(f'Classification report for the support vector machine model: {class_report_svm}')

    print(f'Classification report for the k-nearest neighbor model: {class_report_knn}')


    confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)

    print(f'Confusion matrix for the logistic regression model: {confusion_matrix_lr}')

    print(f'Confusion matrix for the support vector machine model: {confusion_matrix_svm}')

    print(f'Confusion matrix for the k-nearest neighbor model: {confusion_matrix_knn}')
