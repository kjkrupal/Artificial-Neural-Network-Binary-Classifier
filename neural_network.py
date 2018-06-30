import sys
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    return X, y

def encode_data(X):
    labelencoder_x_1 = LabelEncoder()
    X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
    labelencoder_x_2 = LabelEncoder()
    X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
    onehotencoder = OneHotEncoder(categorical_features=[1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    return X

def prepare_train_test_set(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


def train_network(X_train, y_train):
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    return classifier, best_parameters, best_accuracy

def predict_result(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    X, y = load_dataset(dataset_path)
    X = encode_data(X)
    X_train, X_test, y_train, y_test = prepare_train_test_set(X, y)
    X_train, X_test = scale_features(X_train, X_test)
    classifier, best_parameters, best_accuracy = train_network(X_train, y_train)
    print("Best parameters: \n", best_parameters, "\nBest accuracy: \n", best_accuracy)
    conf_matrix = predict_result(X_test, y_test, classifier)
    print("Confusion Matrix: \n", conf_matrix)
