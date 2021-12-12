import pandas as pd
from sklearn.model_selection import train_test_split

features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

target = ['Class']


def load_dataset(path = './data/',test_size = 0.25):
    df = pd.read_csv(path + 'creditcard.csv')
    return train_test_split(df[features],df[target],test_size = test_size, stratify = df[target])    