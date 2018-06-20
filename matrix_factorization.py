import numpy as np

class MatrixFactorization():
    """
    This class receives a training data R, and test data R_test
    and factorize R into user characterstic matrix U and item cahracteristic matrix V
    with bias vectors of users and items.
    
    """

    def __init__(self, R, R_test, user_size, item_size, K=10, a=0.01, b=0.2):
        """
        Input variables:
        - R (np.array) : Rating data. R[i] = [[user_id, item_id, rating]] assumed.
                         The rating score should be dimensionless quantity.
        - R_test : Test data by which we examine the validity of the learning model.
        - K (int) : the size of user vector U[i] and item vector V[j].
        - a (float) : learning rate parameter
        - b (float) : overall coefficient of the regularization terms.
        """

        # Read and store input variables
        self.__R = R
        self.__R_test = R_test
        self.__K = K
        self.__a = a
        self.__b = b

        # Get the size of training and test data
        self.__train_size = self.__R.shape[0]
        self.__test_size = self.__R_test.shape[0]

        # Get the number of user and item.
        self.__user_size = user_size
        self.__item_size = item_size

        # Compute mean rating mu
        self.__mu = np.sum(self.__R[:,2])/self.__train_size

    def sgd(self, iu, ii, Rui):
        """
        Perform stochastic gradient descent for (user, item) = (iu, ii)
        """
        # eui = (obsered rating - predicted rating) of (iu, ii) pair
        eui = Rui \
            - (self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii]))

        # shift parameters by X -> X - a*df/dX, where f = (error)^2 + b*(regularization)
        self.__U[iu] = self.__U[iu] - self.__a*(-2*eui*self.__V[ii] + self.__b*self.__U[iu])
        self.__V[ii] = self.__V[ii] - self.__a*(-2*eui*self.__U[iu] + self.__b*self.__V[ii])
        self.__bu[iu] = self.__bu[iu] - self.__a*(-2*eui + self.__b*self.__bu[iu])
        self.__bi[ii] = self.__bi[ii] - self.__a*(-2*eui + self.__b*self.__bi[ii])

    def error(self):
        """
        Compute (mean square error) + (Regulatization), which is minimized by SGD method.
        """
        error = 0.0

        for r in self.__R:
            iu = int(r[0])-1
            ii = int(r[1])-1
            error = error + (r[2] - (self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii])))**2
        
        # Add regularization terms
        error = error/self.__train_size + self.__b/2*(np.sum(self.__U**2) + np.sum(self.__V**2) + np.sum(self.__bu**2) + np.sum(self.__bi**2))
        
        return error

    def error_test(self):
        """
        Compute sqrt(mean square error) of test data.
        """
        error = 0.0

        for r in self.__R_test:
            iu = int(r[0])-1
            ii = int(r[1])-1
            error = error + (r[2] - (self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii])))**2
        
        return np.sqrt(error/self.__test_size)
        
    def train(self, tol=1.e-2, maxitr=200000, debug=False):
        """
        Training module for a given rating matrix R.

        - tol: relative tolerance. 
        - maxitr: the max number of epoch.
        - debug: if True, intermediate result is printed.
        
        If |error[(n+1)epoch]/error[n epoch]  - 1| < tol, training finishes.
       
        We first prepair the objects which factorize rating matrix R:
        - U = (u_1, u_2, ...)^T: u_i characterizes the user i.
        - V = (v_1, v_2, ...)^T: v_i characterizes the item i.
        - bu: bias vector of user.
        - bi: bias vector of item.
        
        predicted rating matrix takes the form of mu + bu + bi + U.V^T.
        """
        self.__U = np.random.rand(self.__user_size, self.__K)
        self.__V = np.random.rand(self.__item_size, self.__K)
        self.__bu = np.zeros(self.__user_size)
        self.__bi = np.zeros(self.__item_size)

        # initialize the object for np.fabs(new_result / old_result)
        ratio = tol*1000
        # initialize the object for error function
        err_new = self.error()

        # Iteration of sgd
        i_rands = np.arange(self.__train_size)
        if debug is True:
            print("-----------------------------------------------------")
            print("epoch, convergence, error function, RMSE of test data")
        for i in range(maxitr):
            # Shuffle the set of training data
            np.random.shuffle(i_rands)
            for i_rand in i_rands:
                # sgd for randomly chosen (user, item) pair
                iu = int(self.__R[i_rand][0])-1
                ii = int(self.__R[i_rand][1])-1
                Rui = int(self.__R[i_rand][2])
                self.sgd(iu=iu, ii=ii, Rui=Rui)
            
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
