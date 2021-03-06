import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pdb


def compute_y(x, W, bias):
    return (-x * W[0] - bias) / (W[1] + 1e-10)

def plot_decision_boundary(X, y , W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    color = 'r'
    if(current_y == -1):
        color = 'b'
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    # ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color+'s')
    # afisarea dreptei de decizie
    plt.plot([x1, x2] ,[y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)

def compute_accuracy(x, y, w, b):
    accuracy = (np.sign(np.dot(x, w) + b) == y).mean()
    return accuracy

def train_perceptron(x, y, num_epochs, learning_rate):
    num_samples = x.shape[0]
    num_features = x.shape[1]

    w = np.zeros(num_features)
    b = 0
    accuracy = 0.0
    for epoch in range(epochs):
        # 4.1
        x, y = shuffle(x, y)

        # 4.2
        for i in range(num_samples):
            y_hat = np.dot(x[i, :], w) + b
            loss = (y_hat - y[i]) ** 2
            w = w - lr * (y_hat - y[i]) * x[i, :]
            b = b - lr * (y_hat - y[i])
            accuracy = compute_accuracy(x, y, w, b)

            print("epoch: ", epoch, " sample: ", i," sample_loss: ", loss, " accuracy: ", accuracy)
            plot_decision_boundary(x, y, w, b, x[i, :], y[i])
    return w, b, accuracy

x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y = np.array([-1, 1, 1, 1])

#x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
#y = np.array([-1, 1, 1, -1])

epochs = 70
lr = 0.1

w, b, accuracy = train_perceptron(x, y, epochs, lr)
print("weight: ", w, " bias: ", b, " accuracy: ", accuracy)

