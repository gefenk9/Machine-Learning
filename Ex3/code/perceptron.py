#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sgd import helper


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels):
    """
	returns: nd array of shape (data.shape[1],)
	or (data.shape[1],1) representing the perceptron classifier
    """
    w = np.zeros(len(data[0]))
    for x in range(len(data)):
        normalized = data[x] / np.linalg.norm(data[x])
        pred = np.dot(w, normalized)
        label = labels[x]
        if pred >= 0 and label == -1:
            w = [w_i - x_i for w_i, x_i in zip(w, normalized)]
        if pred < 0 and label == 1:
            w = [w_i + x_i for w_i, x_i in zip(w, normalized)]

    return w

def _calc_accuracy(w, test_data, test_labels):
    suc = 0.0
    for i in range(len(test_data)):
        pred = np.dot(w, test_data[i])
        if pred >= 0 and test_labels[i] == 1 or pred < 0 and test_labels[i] == -1:
            suc += 1.0
    return suc/len(test_data)

#################################

# Place for additional code

#################################
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
# 1.a


table = {}
w = None
for n in [5,10,50,100,500,1000, 5000]:
    table[n] = []
    accuracy = [-1.0]*100
    try:
        for run in range(100):
            p = np.random.permutation(n)
            train_labels_sample = train_labels[p]
            train_data_smaple = train_data[p]
            w = perceptron(train_data_smaple[:n], train_labels_sample[:n])
            accuracy_prec = _calc_accuracy(w, test_data, test_labels)
            accuracy[run] = accuracy_prec
    except Exception as e:
        print (e.message, n)
    table[n].extend([(sum(accuracy)/100)*100, accuracy[4]*100, accuracy[94]*100])
#1.a
for n in table:
    print ("n\tmean\t5th precent\t 95th precent")
    print ("{}\t{}\t{}\t{}".format(n, table[n][0], table[n][1], table[n][2]))
#1.b
w = perceptron(train_data, train_labels)
plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
plt.savefig('classifier.png')
plt.close()
#1.c
accuracy = _calc_accuracy(w, test_data, test_labels)
print (accuracy)
#1.d
for i in range(len(test_data)):
    pred = np.dot(w, test_data[i])
    if pred >= 0 and test_labels[i] == -1 or pred < 0 and test_labels[i] == 1:
        mean_of_array = test_data[i].mean(axis=0)
        std_of_array = test_data[i].std(axis=0)
        unscaled = (test_data[i] * std_of_array) + mean_of_array
        plt.imshow(np.reshape(unscaled, (28, 28)), interpolation='nearest')
        plt.savefig('misclassified_test.png')
        plt.close()
