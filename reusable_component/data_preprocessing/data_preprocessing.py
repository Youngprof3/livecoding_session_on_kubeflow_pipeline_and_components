
import pandas as pd
import argparse
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def data_preprocessing():
    with open("data.pkl","rb") as f:
        data = pickle.load(f)
        #data features
        X = data.iloc[:,:-1]
        #target data
        y = data.iloc[:,-1:]
        #encoding the categorical columns
        le = LabelEncoder()

        X['gender'] = le.fit_transform(X['gender'])
        y['class'] = le.fit_transform(y['class'])

        #splitting the data
        X_train,X_test,y_train,y_test = train_test_split( X,y, test_size=0.3, random_state = 42)
        #feature scaling
        ms =MinMaxScaler(feature_range=(0,1))
        X_train = ms.fit_transform(X_train)
        X_test = ms.transform(X_test)

        #Save the train_data as a pickle file to be used by the train component.
        with open('train_data.pkl', 'wb') as f:
            pickle.dump((X_train,  y_train), f)

        #Save the test_data as a pickle file to be used by the predict component.
        with open('test_data.pkl', 'wb') as f:
            pickle.dump((X_test,  y_test), f)



if __name__ == '__main__':
    data_preprocessing()