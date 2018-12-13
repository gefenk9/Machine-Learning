#################################
# Your name: Gefen keinan
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    support_vectors = np.zeros(shape=(3,2))
    C = 1000

    #linear
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    support_vectors[0] = svc.n_support_
    create_plot(X_train, y_train, svc)
    plt.show()

    #quadratic
    svc = svm.SVC(kernel='poly', C=C).fit(X_train, y_train)
    support_vectors[1] = svc.n_support_
    create_plot(X_train, y_train, svc)
    plt.show()

    #rbf
    svc = svm.SVC(kernel='rbf', C=C).fit(X_train, y_train)
    support_vectors[2] = svc.n_support_
    create_plot(X_train, y_train, svc)
    plt.show()

    return support_vectors

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    results = {}
    for exp in range(-5, 6):
        C = 10 ** exp
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)

        # validate the model
        suc_val = 0
        predicted = svc.predict(X_val)
        for i in range(len(predicted)):
            if predicted[i] == y_val[i]:
                suc_val += 1.0
        results[C] = [suc_val / len(y_val), svc.score(X_train, y_train)]
        create_plot(X_train, y_train, svc)
        plt.savefig('.\C\{} boundries.png'.format(C))
    return results

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C = 10
    results = {}
    for exp in range(-5, 6):
        gamma = 10 ** exp
        svc = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X_train, y_train)

        # validate the model
        suc_val = 0
        predicted = svc.predict(X_val)
        for i in range(len(predicted)):
            if predicted[i] == y_val[i]:
                suc_val += 1.0
        results[gamma] = [suc_val / len(y_val), svc.score(X_train, y_train)]
        create_plot(X_train, y_train, svc)
        plt.savefig('.\gamma\{} boundries-gamma.png'.format(gamma))
    return results



train_x, train_y, val_x, val_y = get_points()
#support_vectors = train_three_kernels(train_x, train_y, val_x, val_y)
#results = linear_accuracy_per_C(train_x, train_y, val_x, val_y)
results = rbf_accuracy_per_gamma(train_x, train_y, val_x, val_y)