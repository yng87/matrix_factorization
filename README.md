# matrix_factorization
This module provides MatrixFactorization class, a symple implementation of the matrix factorization for a recommendation system.

It works on Python3.

## Getting Started

### Installing

Download and unzip. Then install by
```
cd matrix_factorization
pip install ./
```

### How to use
To use in your code, import the module by, e.g., 
```
from matrix_factorization import MatrixFactorization
```
then give training rating data ```R```, test data ```R_test```, 
and the number of users (items) ``` user_size``` (```item_size```) and perform training
```
mf = MatrixFactorization(R, R_test=, user_size, item_size)
mf.train()
```
Note that the i-th element of ```R``` and ```R_test``` is assumed to be ```[user ID, item ID, rating]```.
The resulting rating matrix is called by
```
mf.rating_matrix()
```

## Running the test

In the directory ```tests```, there is a test code ```test.py```.
The data directory ```ml-100k```is taken from https://grouplens.org/datasets/movielens/.
It offers the ratings of movies by 1, 2, 3, 4 or 5.

Run the test code by
```
python test.py
```
You can provide the following command line arguments:
```
--data_dir (default="ml-100k/"): data directory path
--base (default="u1.base"): dataset for training
--test (default="u1.test"): dataset for test
--K (default=50): The row size of U and V for R = U^T V
--a (default=0.01): learning rate parameter
--b (default=0.2): coefficient of regularization terms
--tol (defalut=3.e-2): requirement for convergence
```

