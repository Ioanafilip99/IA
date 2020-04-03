from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

from sklearn.neural_network import MLPClassifier  # importul clasei
from sklearn import preprocessing

import numpy as np

def plot3d_data(X, y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[ y == -1, 0 ], X[ y == -1, 1 ], X[ y == -1, 2 ], 'b');
    ax.scatter3D(X[ y == 1, 0 ], X[ y == 1, 1 ], X[ y == 1, 2 ], 'r');
    plt.show()


def plot3d_data_and_decision_function(X, y, W, b):
    ax = plt.axes(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    # calculate corresponding z
    # [x, y, z] * [coef1, coef2, coef3] + b = 0
    zz = (-W[ 0 ] * xx - W[ 1 ] * yy - b) / W[ 2 ]
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.scatter3D(X[ y == -1, 0 ], X[ y == -1, 1 ], X[ y == -1, 2 ], 'b');
    ax.scatter3D(X[ y == 1, 0 ], X[ y == 1, 1 ], X[ y == 1, 2 ], 'r');
    plt.show()


# incarcarea datelor de antrenare
X = np.loadtxt('./data/3d-points/x_train.txt')
y = np.loadtxt('./data/3d-points/y_train.txt', 'int')
y.astype(int)

# plot3d_data(X, y)
# incarcarea datelor de testare
X_test = np.loadtxt('./data/3d-points/x_test.txt')
y_test = np.loadtxt('./data/3d-points/y_test.txt', 'int')

sc = preprocessing.StandardScaler()
sc.fit(X)
sc.fit(X_test)
X_sc = sc.transform(X)
X_test_sc= sc.transform(X_test)

perceptron_model = Perceptron(eta0=0.1, tol=1e-5)

perceptron_model.fit(X_sc,y)
print("Acuratete pe multimea de antrenare: ", perceptron_model.score(X_sc, y))
print("Acuratete pe multimea de testare: ", perceptron_model.score(X_test_sc, y_test))

plot3d_data_and_decision_function(X_sc, y, np.squeeze(perceptron_model.coef_), perceptron_model.intercept_)

plot3d_data_and_decision_function(X_test_sc, y_test, np.squeeze(perceptron_model.coef_), perceptron_model.intercept_)

print(perceptron_model.n_iter_)
print(perceptron_model.coef_)
print(perceptron_model.intercept_)
