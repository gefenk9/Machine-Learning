#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper():
  mnist = fetch_mldata('MNIST original')
  data = mnist['data']
  labels = mnist['target']

  neg, pos = 0,8
  train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
  test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

  train_data_unscaled = data[train_idx[:6000], :].astype(float)
  train_labels = (labels[train_idx[:6000]] == pos)*2-1

  validation_data_unscaled = data[train_idx[6000:], :].astype(float)
  validation_labels = (labels[train_idx[6000:]] == pos)*2-1

  test_data_unscaled = data[60000+test_idx, :].astype(float)
  test_labels = (labels[60000+test_idx] == pos)*2-1

  # Preprocessing
  train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
  validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
  test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
  return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """
    w = np.zeros(len(data[0]))
    for i in range(min(len(data), T)):
        eta = eta_0 / (i + 1)
        x = data[i]
        label = labels[i]
        h = np.dot(w, x)
        w_true = [w_i * (1 - eta) for w_i in w]
        if label*h < 1:
            w = [w_i + eta * C * label * x_i for w_i, x_i in zip(w_true, x)]
        else:
            w = w_true
    return w
#################################

# Place for additional code

#################################
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

#2.a
teta_suc = []
teta_l = []
times = 10
for scale in range(-5,6):
    teta = 10**scale
    teta_l.append(teta)
    C=1
    T = 1000
    suc = []
    for t in range(T):
        p = np.random.permutation(len(train_data))
        train_labels_sample = train_labels[p]
        train_data_sample = train_data[p]
        w = SGD(train_data_sample, train_labels_sample, C, teta, T)

        suc.append(0.0)
        for i in range(len(validation_data)):
            h = np.dot(w, validation_data[i])
            if validation_labels[i] * h >= 1:
                suc[t] += 1.0
        suc[t] /= float(len(validation_data))

    teta_suc.append(float(sum(suc)) / times)

# plotting the  performance of η0 by averaging the accuracy on the validation set
# across 10 runs.

plt.plot(teta_l, teta_suc)
plt.ylabel('accuracy')
plt.xlabel('eta_0')
plt.xscale('log')
plt.xlim(10**(-5),10**5)
plt.savefig('eta.png')
plt.close()
best_eta = teta_l[teta_suc.index(max(teta_suc))]
print ("best eta_0 is {}".format(best_eta))

#2.b

C_success = []
C_list = []

for j in range(-5, 6):
    C_j = 10 ** j
    C_list.append(C_j)

    suc = []
    for t in range(times):
        p = np.random.permutation(len(train_data))
        train_labels_sample = train_labels[p]
        train_data_sample = train_data[p]

        w = SGD(train_data_sample, train_labels_sample, C_j, best_eta, T)

        suc.append(0.0)
        for i in range(len(validation_data)):
            h = np.dot(w, validation_data[i])
            if validation_labels[i] * h >= 1:
                suc[t] += 1.0
        suc[t] /= float(len(validation_data))

    C_success.append(float(sum(suc)) / times)

plt.plot(C_list, C_success)
plt.ylabel('accuracy')
plt.xlabel('C')
plt.xscale('log')
plt.xlim(10 ** (-5), 10 ** 5)
plt.ylim(0.9, 1)
plt.savefig('C.png')
plt.close()

best_c = C_list[C_success.index(max(C_success))]
print ("best C is {}".format(best_c))

