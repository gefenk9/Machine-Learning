#################################
# Your name:gefen keinan
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(X_train)
    alphas = []
    hypothesises = []

    # Initialize distribution weights
    Dt = np.array([1.0 / n for i in range(n)])
    for t in range(T):
        # Calculate erm_decision_stump and its error foreach class A and B
        hypothesis_A, error_A = erm_decision_stump(X_train, y_train, Dt, 1)
        hypothesis_B, error_B = erm_decision_stump(X_train, y_train, Dt, np.negative(1))
        if error_A < error_B:
            hypothesis_t = hypothesis_A
            error_t = error_A
        else:
            hypothesis_t = hypothesis_B
            error_t = error_B

        alpha_t = 0.5 * np.log((1 - error_t) / error_t)

        hypothesises.append(hypothesis_t)
        alphas.append(alpha_t)

        # Update distribution
        softmax_vals = np.array(
            [np.exp(np.negative(alpha_t * y_train[j] * hypothesis_t[0](X_train[j]))) for j in range(n)])
        Zt = np.dot(Dt, softmax_vals)
        new_softmax_vals = softmax_vals * (1.0 / Zt)
        Dt = np.multiply(Dt, new_softmax_vals)

        curr_boosted_h = lambda x: np.sign(np.sum([(hypothesises[j][0](x) * alphas[j]) for j in range(t + 1)]))
        print ("Error decreased to:", calculate_error(X_train, y_train, curr_boosted_h))

    return hypothesises, alphas



##############################################
# You can add more methods here, if needed.

def erm_decision_stump(X_train, y_train, D, b):
    m = len(X_train)
    d = len(X_train[0])

    labeled_data = list(zip(X_train, y_train, D))
    x_index,y_index,D_index = 0,1,2

    best_f = float('inf')
    theta = X_train[0][0] - 1
    best_j = 0

    for j in range(d):
        sorted_by_j = sorted(labeled_data, key=lambda labeled_point: labeled_point[x_index][j])

        aprox_last_labeled_point = [(sorted_by_j[m - 1][x_index][j] + 1) for i in range(d)]
        sorted_by_j.append((aprox_last_labeled_point, float('inf'), float('inf')))
        curr_f = np.sum([labeled_point[D_index] for labeled_point in sorted_by_j if labeled_point[y_index] == b])

        if curr_f < best_f:
            best_f = curr_f
            theta = sorted_by_j[0][x_index][j] - 1
            best_j = j

        # Sorted points iteration
        for i in range(m):
            curr_f = curr_f - (b * (sorted_by_j[i][y_index] * sorted_by_j[i][D_index]))

            if (curr_f < best_f) and (sorted_by_j[i][x_index][j] != sorted_by_j[i + 1][x_index][j]):
                best_f = curr_f
                theta = 0.5 * (sorted_by_j[i][x_index][j] + sorted_by_j[i + 1][x_index][j])
                best_j = j

    return (lambda x: np.sign(theta - x[best_j]) * b, best_j, theta), best_f

def calculate_error(data, labels, hypothesis):
    error = 0
    for i in range(len(data)):
        if hypothesis(data[i]) != labels[i]:
            error += 1
    return float(error) / len(data)


##############################################


def b(X_train,y_train, vocab):
    hypotheses, alphas = run_adaboost(X_train, y_train, T=10)
    for t in range(10):
        print(vocab[hypotheses[t][1]])

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    b(X_train,y_train,vocab)

    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)


    train_errors = []
    test_errors = []
    train_func_l = []
    test_func_l  = []
    for t in range(T):
        curr_boosted_h = lambda x: np.sign(
            np.sum([(hypotheses[i][0](x) * alpha_vals[i]) for i in range(t + 1)]))

        train_errors.append(calculate_error(X_train, y_train, curr_boosted_h))
        test_errors.append(calculate_error(X_test, y_test, curr_boosted_h))

        # For section C
        L_softmax_vals_func = lambda Xi,Yi:\
            [np.exp(np.negative(Yi * np.sum([alpha_vals[j] * hypotheses[j][0](Xi) for j in range(t)])))]
        L_func = lambda X,Y: 1 / len(X) * np.sum([L_softmax_vals_func(X[i], Y[i]) for i in range(len(X))])
        train_func_l.append(L_func(X_train, y_train))
        test_func_l.append(L_func(X_test, y_test))



    # 1(a)
    plt.figure()
    plt.title("errors per t")
    train_error_points, = plt.plot(np.arange(1, T + 1, 1), train_errors,  label='Train Errors')
    test_error_points, = plt.plot(np.arange(1, T + 1, 1), test_errors,  label='Test Errors')
    plt.legend(handles=[train_error_points, test_error_points])
    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.axis([0, 85, 0, 0.6])
    plt.xticks(np.arange(0, 85, 10))
    plt.yticks(np.arange(0, 0.6, 0.05))
    plt.savefig('1(a).png')

    #  1(c)
    plt.figure()
    plt.title("test train loss , # of WL")
    train_func_l_points, = plt.plot(np.arange(1, T + 1, 1), train_func_l,  label='train loss')
    test_func_l_points, = plt.plot(np.arange(1, T + 1, 1), test_func_l,  label='test loss')
    plt.legend(handles=[train_func_l_points, test_func_l_points])
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('loss')
    plt.axis([0, 85, 0, 1.1])
    plt.xticks(np.arange(0, 85, 10))
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.savefig('1(c).png')

    ##############################################

if __name__ == '__main__':
    main()