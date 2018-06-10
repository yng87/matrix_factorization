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

        #Compute mean rating mu
        self.__mu = np.sum(self.__R)/np.sum(self.__R > 0)

    def sgd(self, iu, ii):
        eui = self.__R[iu,ii] \
            - (self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii]))
        self.__U[iu] = self.__U[iu] - self.__a*(-2*eui*self.__V[ii] + self.__b*self.__U[iu])
        self.__V[ii] = self.__V[ii] - self.__a*(-2*eui*self.__U[iu] + self.__b*self.__V[ii])
        self.__bu[iu] = self.__bu[iu] - self.__a*(-2*eui + self.__b*self.__bu[iu])
        self.__bi[ii] = self.__bi[ii] - self.__a*(-2*eui + self.__b*self.__bi[ii])

    def error(self):
        error = 0.0
        ius, iis = self.__R.nonzero()
        for iu, ii in zip(ius, iis):
            error = error + (self.__R[iu, ii] - (self.__mu + self.__bu[iu] + self.__bi[ii] + np.dot(self.__U[iu], self.__V[ii])))**2
            error = error + self.__b/2*(np.sum(self.__U[iu]**2) + np.sum(self.__V[ii]**2) + self.__bu[iu]**2 + self.__bi[ii]**2)
        return np.sqrt(error)
        
    def train(self, tol=1.0, step=1000, debug=True):
        self.__U = np.random.rand(self.__user_size, self.__K)
        self.__V = np.random.rand(self.__item_size, self.__K)
        self.__bu = np.zeros(self.__user_size)
        self.__bi = np.zeros(self.__item_size)

        iiter = 0
        diff = tol*1000
        err_new = self.error()
        ius, iis = self.__R.nonzero()
        while diff > tol:
            i_rand = np.random.randint(0, np.size(ius))
            self.sgd(iu=ius[i_rand], ii=iis[i_rand])
            if iiter % step ==0:
                err_old = err_new
                err_new = self.error()
                diff = np.fabs(err_new - err_old)
                if debug is True:
                    print(iiter, err_new)
            iiter = iiter + 1
        if debug is True:
            print("Total iteration is ", iiter, "Resultant mean square error is ", err_new)

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
            
if __name__ == "__main__":
    mf = MatrixFactorization(np.array([[1,0, 0, 4], [0,3,2,2], [3,4,0,0]]), 10)
    mf.train(tol=1.e-3, step=1000)
    print(mf.U, mf.V, mf.bu, mf.bi)
