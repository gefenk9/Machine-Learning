#################################
# Your name:
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
    # TODO: add your code here
    D =[]
    m = len(X_train)
    D_0 = [1/m]*m #First disrebution
    D.append(D_0)
    for i in range(T):
        h_t, epsilon = _pick_best_learner(D[i], X_train, y_train)
        a_t = 0.5*np.log((1-epsilon)/epsilon)
        D_next = []

        normalized_factor = 0
        for z in range(m):
            normalized_factor += D[i][z] * np.exp(-a_t * y_train[z] * h_t(X_train[z]))
        for j in range(m):
            d_next_j = (D[i][j] * np.exp(-a_t*y_train[j]*h_t(X_train[j]))) / normalized_factor
            D_next.append(d_next_j)


##############################################
# You can add more methods here, if needed.

def _pick_best_learner():
    pass





##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T=80)

    ##############################################
    # You can add more methods here, if needed.



    ##############################################

if __name__ == '__main__':
    main()



