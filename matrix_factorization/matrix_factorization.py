import numpy as np
import pandas as pd

class MatrixFactorization():
    """
    This class receives a matrix M which contains rating data of the form 
    R[user_id, item_id] = rating of item[item_id] by user[user_id],
    and factorize M into user characterstic matrix U and item cahracteristic matrix V;
    R = U.V with bias vectors of users (bu) and items (bv).
    """

    def __init__(self, R, K, a=2.e-4, b=0.02):
        """
        Input variables:
        - R (np.array) : Rating data R[user_id, item_id] used for training.
                         The rating score should be 1, 2, 3, 4 or 5.
                         Missing data should be set at 0.
        - K (int) : the size of user vector U[i] and item vector V[j].
        - a (float) : parameter for optimization. 
        - b (float) : parameter for regularization.
        """
        
        self.__R = R
        self.__K = K
        self.__a = a
        self.__b = b

        # Get the number of user and that of item.
        self.__user_size = self.__R.shape[0]
        self.__item_size = self.__R.shape[1]

        print(self.__user_size, self.__item_size)
        

if __name__ == "__main__":
    mf = MatrixFactorization(np.array([[1,2, 3, 4], [2,3,2,2], [3,4,2,2]]), 10)
