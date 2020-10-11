from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def heart_failure_prediction():
    data = pd.read_csv('data/heart-failure-prediction.csv')
    columns = data.columns.drop('DEATH_EVENT')
    x = pd.DataFrame(data, columns=columns)
    y = pd.DataFrame(data, columns=['DEATH_EVENT'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    return x_train, y_train, x_test, y_test

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    return scaler.transform(x_train), scaler.transform(x_test)

def encode_data(y_train, y_test):
    encoder = OneHotEncoder()
    e_train = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).todense()
    e_test = encoder.transform(y_test.to_numpy().reshape(-1, 1)).todense()

    return e_train, e_test