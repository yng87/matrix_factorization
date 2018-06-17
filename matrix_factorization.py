import numpy as np

class MatrixFactorization():
    """
    This class receives a matrix R which contains rating data of the form 
    R[user_id, item_id] = rating of item[item_id] by user[user_id],
    and factorize R into user characterstic matrix U and item cahracteristic matrix V;
    R = UV with bias vectors of users (bu) and items (bv).
    """

    def __init__(self, R, R_test, K=10, a=0.01, b=0.2):
        """
        Input variables:
        - R (np.array) : Rating data R[user_id, item_id] used for training.
                         The rating score should be 1, 2, 3, 4 or 5.
                         Missing data should be set at 0.
        - R_test : Rating data for test
        - K (int) : the size of user vector U[i] and item vector V[j].
        - a (float) : parameter for optimization. 
        - b (float) : parameter for regularization.
        """
        
        self.__R = R
        self.__R_test = R_test
        self.__K = K
        self.__a = a
        self.__b = b

        self.__train_size = np.sum(self.__R > 0)
        self.__test_size = np.sum(self.__R_test > 0)

        # Get the number of user and of item.
        self.__user_size = self.__R.shape[0]
        self.__item_size = self.__R.shape[1]

        # Compute mean rating mu
        self.__mu = np.sum(self.__R)/self.__train_size

        # Stores obsered element of R as 1, and missing element as 0
        self.__obs_index = (self.__R != 0).astype(int)
        self.__test_index = (self.__R_test != 0).astype(int)

    def sgd(self, iu, ii):
        """
        Perform stochastic gradient descent for (user, item) = (iu, ii)
        """
        # eui = (obsered rating - predicted rating) of (iu, ii) pair
        eui = self.__R[iu,ii] \
            - (self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii]))

        # shift parameters by X -> X - a*df/dX, where f = (error)^2 + b*(regularization)
        self.__U[iu] = self.__U[iu] - self.__a*(-2*eui*self.__V[ii] + self.__b*self.__U[iu])
        self.__V[ii] = self.__V[ii] - self.__a*(-2*eui*self.__U[iu] + self.__b*self.__V[ii])
        self.__bu[iu] = self.__bu[iu] - self.__a*(-2*eui + self.__b*self.__bu[iu])
        self.__bi[ii] = self.__bi[ii] - self.__a*(-2*eui + self.__b*self.__bi[ii])

    def error(self):
        """
        Compute sqrt(mean squared error), which is minimized by sgd method.
        """
        error = 0.0

        # Rhat is predicted rating matrix
        Rhat = self.__mu + self.__bu[:, np.newaxis] + self.__bi[np.newaxis, :] + np.dot(self.__U, np.transpose(self.__V))
        
        # Squared error consisting only of the observed element of rating matrix R.
        error = np.sum(((self.__R - Rhat) * self.__obs_index)**2)/self.__train_size
        
        # Add regularization terms
        error = error + self.__b/2*(np.sum(self.__U**2) + np.sum(self.__V**2) + np.sum(self.__bu**2) + np.sum(self.__bi**2))
        
        return error

    def error_test(self):
        """
        Compute sqrt(mean squared error).
        """
        error = 0.0

        # Rhat is predicted rating matrix
        Rhat = self.__mu + self.__bu[:, np.newaxis] + self.__bi[np.newaxis, :] + np.dot(self.__U, np.transpose(self.__V))
        
        # Squared error consisting only of the observed element of rating matrix R.
        error = np.sum(((self.__R_test - Rhat) * self.__test_index)**2)/self.__test_size
        
        # Add regularization terms
        #error = error + self.__b/2*(np.sum(self.__U**2) + np.sum(self.__V**2) + np.sum(self.__bu**2) + np.sum(self.__bi**2))
        
        return np.sqrt(error)
        
    def train(self, tol=1.e-2, maxitr=2000000, debug=False):
        """
        Training module for a given rating matrix R.

        - tol: relative tolerance. 
        - maxitr: the max number of epoch.
        - debug: if True, intermediate result is printed.
        
        If |error[(n+1)epoch]/error[n epoch]  - 1| < atol, training finishes.
       
        We first prepair the objects which factorize rating matrix R:
        - U = (u_1, u_2, ...)^T: u_i characterizes the preference of user i.
        - V = (v_1, v_2, ...)^T: v_i characterizes the item i.
        - bu: bias vector of user.
        - bi: bias vector of item.
        
        predicted rating matrix = mu + bu + bi + U.V^T.
        """
        self.__U = np.random.rand(self.__user_size, self.__K)
        self.__V = np.random.rand(self.__item_size, self.__K)
        self.__bu = np.zeros(self.__user_size)
        self.__bi = np.zeros(self.__item_size)

        # initialize the object for np.fabs(new_result / old_result)
        ratio = tol*1000
        # initialize the object for sqrt(squared error)
        err_new = self.error() 

        # The set of [user_id], [item_id] giving nonzero rating
        ius, iis = self.__R.nonzero()

        # Iteration of sgd
        size_nonzero = np.size(ius)
        i_rands = np.arange(size_nonzero)
        if debug is True:
            print("-----------------------------------------------------")
            print("epoch, convergence, error function, RMSE of test data")
        for i in range(maxitr):
            # Shuffle the set of training data
            np.random.shuffle(i_rands)
            for i_rand in i_rands:
                # sgd for randomly chosen (user, item) pair
                self.sgd(iu=ius[i_rand], ii=iis[i_rand])

            err_old = err_new
            err_new = self.error()
            ratio = err_new/err_old
            if debug is True: print(i, 1-ratio, err_new, self.error_test())
            if np.fabs(ratio - 1.0) < tol:break
        if debug is True:
            print("-----------------------------------------------------")

    @property
    def U(self):
        return self.__U

    @property
    def V(self):
        return self.__V

    @property
    def bu(self):
        return self.__bu

    @property
    def bi(self):
        return self.__bi

    def rating(self, iu, ii):
        """
        Return predicted rating of item ii by user iu.
        """
        return self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii])

    def rating_matrix(self):
        """
        Return whole predicted rating matrix.
        """
        return self.__mu + self.__bu[:, np.newaxis] + self.__bi[np.newaxis, :] + np.dot(self.__U, np.transpose(self.__V))
            
if __name__ == "__main__":
    mf = MatrixFactorization(np.array([[1,0, 0, 4], [0,3,2,2], [3,4,0,0]]), 10, a=0.01)
    mf.train(atol=1.e-4, step=1000, debug=True)
    #print(mf.U, mf.V, mf.bu, mf.bi)
    print(mf.rating(0,1))
    print(mf.rating_matrix())
