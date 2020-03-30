import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pdb
import matplotlib.pyplot as plt

# y_hat = sigmoid(tanh(X * W_1 + b_1) * W_2 + b_2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def forward(x, W1, b_1, W2, b_2):
    z_1 = np.matmul(x, W1) + b_1
    a_1 = np.tanh(z_1)

    z_2 = np.matmul(a_1, W2) + b_2
    a_2 = sigmoid(z_2)
    return z_1, a_1, z_2, a_2

def backward(a_1, a_2, z_1, W_2, X, Y, num_samples):
    dz_2 = a_2 - y # derivata functiei de pierdere (logistic loss) in functie de z
    dw_2 = np.matmul(a_1.T, dz_2) / num_samples # np.dot
    # der(L/w_2) = der(L/z_2) * der(dz_2/w_2) = dz_2 * der((a_1 * W_2 + b_2)/ W_2)
    db_2 = np.sum(dz_2, axis= 0) / num_samples  # np.sum
    # der(L/b_2) = der(L/z_2) * der(z_2/b_2) = dz_2 * der((a_1 * W_2 + b_2)/ b_2)
    # primul strat
    da_1 = np.matmul(dz_2,W_2.T) # np.dot
    # der(L/a_1) = der(L/z_2) * der(z_2/a_1) = dz_2 * der((a_1 * W_2 + b_2)/ a_1)
    dz_1 = np.multiply(da_1, tanh_derivative(z_1))
    # der(L/z_1) = der(L/a_1) * der(a_1/z1) = da_1 .* der((tanh(z_1))/ z_1)
    dw_1 = np.matmul(X.T,dz_1) / num_samples
    # der(L/w_1) = der(L/z_1) * der(z_1/w_1) = dz_1 * der((X * W_1 + b_1)/ W_1)
    db_1 = np.sum(dz_1, axis = 0) / num_samples
    # der(L/b_1) = der(L/z_1) * der(z_1/b_1) = dz_1 * der((X * W_1 + b_1)/ b_1)
    return dw_1, db_1, dw_2, db_2

def compute_y(x, W, bias):
 # dreapta de decizie
 # [x, y] * [W[0], W[1]] + b = 0
    return (-x*W[0] - bias) / (W[1] + 1e-10)

def plot_decision(X_, W_1, W_2, b_1, b_2):
 # sterge continutul ferestrei
    plt.clf()
 # ploteaza multimea de antrenare
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block = False)
    plt.pause(0.1)

X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
print('X.shape = ', X.shape)
y = np.expand_dims(np.array([0, 1, 1, 0]), 1) # [[0], [1], ..]
print('y.shape = ', y.shape)

no_hidden_neurons = 5
no_output_neurons = 1

W_1 = np.random.normal(0, 1, (2, no_hidden_neurons))
b_1 = np.zeros(no_hidden_neurons)
W_2 = np.random.normal(0, 1, (no_hidden_neurons, no_output_neurons))
b_2 = np.zeros(no_output_neurons)

num_samples = X.shape[0]

num_epochs = 70
lr = 0.5
for epoch_idx in range(num_epochs):
    X, y = shuffle(X, y)

    z_1, a_1, z_2, a_2 = forward(X, W_1, b_1, W_2, b_2)

    loss = (-y * np.log(a_2) - (1 - y) * np.log(1 - a_2)).mean()
    accuracy = (np.round(a_2) == y).mean() #hint: a_2
    print("epoch: ", epoch_idx, "loss: ", loss, "accuracy: ", accuracy)
    plot_decision(X, W_1, W_2, b_1, b_2)

    dw_1, db_1, dw_2, db_2 = backward(a_1, a_2, z_1, W_2, X, y, num_samples)
    W_1 = W_1 - lr * dw_1
    b_1 = b_1 - lr * db_1
    W_2 = W_2 - lr * dw_2
    b_2 = b_2 - lr * db_2
