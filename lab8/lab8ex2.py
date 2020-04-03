import numpy as np

from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

train_feats = np.loadtxt('data/MNIST/train_images.txt')
train_lbls = np.loadtxt('data/MNIST/train_labels.txt', 'int')

test_feats = np.loadtxt('data/MNIST/test_images.txt')
test_lbls = np.loadtxt('data/MNIST/test_labels.txt', 'int')

scaler = preprocessing.StandardScaler()

scaler.fit(train_feats)
train_feats_sc = scaler.transform(train_feats)
test_feats_sc = scaler.transform(test_feats)


def train_test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print('Accuracy on train set is: ', model.score(X_train, y_train))
    print('Accuracy on test set is: ', model.score(X_test, y_test))
    print('Number of iterations until convergence is: ', model.n_iter_)

    print('\n')


# a
clf = MLPClassifier(hidden_layer_sizes=(1,),
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# b
clf = MLPClassifier(hidden_layer_sizes=(10,),
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# c
clf = MLPClassifier(hidden_layer_sizes=(10,),
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.00001,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# d
clf = MLPClassifier(hidden_layer_sizes=(10,),
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=10,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# e
clf = MLPClassifier(hidden_layer_sizes=(10,),
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0,
                    max_iter=20)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# f
clf = MLPClassifier(hidden_layer_sizes=(10, 10),
                    activation='tanh',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# g
clf = MLPClassifier(hidden_layer_sizes=(10, 10),
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# h
clf = MLPClassifier(hidden_layer_sizes=(100, 100),
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# i
clf = MLPClassifier(hidden_layer_sizes=(100, 100),
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0.9,
                    max_iter=2000)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)

# j
clf = MLPClassifier(hidden_layer_sizes=(100, 100),
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    momentum=0.9,
                    max_iter=2000,
                    alpha=0.005)
train_test_model(clf, train_feats_sc, train_lbls, test_feats_sc, test_lbls)
