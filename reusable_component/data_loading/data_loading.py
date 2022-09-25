
import os
import argparse
import pandas as pd
import pickle
def data_loading():
    '''
    Function for loading diabetes dataset
    '''
     #reading the data from its source
    data = pd.read_csv("https://raw.githubusercontent.com/Youngprof3/kubeflow_test/main/project_on_diabetes_kubeflow/data_loading/dataset.csv")
     #Save the data as a pickle file to be used by the preprocess component.
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)



if __name__ == '__main__':
    data_loading()