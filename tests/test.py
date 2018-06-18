import pandas as pd
import numpy as np
import sys
import argparse
from  matrix_factorization import MatrixFactorization

def get_rating(data_dir, base, test, debug=False):
    """
    This module read the movie rating data, and perform matrixfactorization.
    """

    # Import data
    train_data = pd.read_table(data_dir+base, delimiter="\t", header=None)
    test_data = pd.read_table(data_dir+test, delimiter="\t", header=None)
    item = pd.read_table(data_dir+"u.item", sep="|", header=None, encoding='latin-1')
    user = pd.read_table(data_dir+"u.user", delimiter="|", header=None)

    # Name each column
    train_data.columns = ["id_user", "id_item", "rating", "timestamp"]
    test_data.columns = ["id_user", "id_item", "rating", "timestamp"]
    item.columns = ["id", "title", "release date", "video release date", "url", "unknown", 
                "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
               "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi",
               "thriller", "war", "western"]
    user.columns = ["id", "age", "gender", "occupation", "zip"]

    # Get the number of users and items
    user_size = np.size(user["id"])
    item_size = np.size(item["id"])

    # Prepare matrix to store rating data for training and test
    R = np.zeros((user_size, item_size))
    R_test = np.zeros((user_size, item_size))

    # Read base data
    if debug is True: print("Getting base data from {}...".format(base))
    for i, row in train_data.iterrows():
        R[row["id_user"]-1, row["id_item"]-1] = row["rating"]

    # Read test data
    if debug is True: print("Getting test data from {}...".format(test))
    for i, row in test_data.iterrows():
        R_test[row["id_user"]-1, row["id_item"]-1] = row["rating"]

    return R, R_test

def main():
    """
    main function for test
    """

    # For command line arguments
    psr = argparse.ArgumentParser()
    psr.add_argument("--data_dir", default="ml-100k/")
    psr.add_argument("--base", default="u1.base")
    psr.add_argument("--test", default="u1.test")
    psr.add_argument("--a", default=0.01, type=float)
    psr.add_argument("--b", default=0.2, type=float)
    psr.add_argument("--K", default=50, type=int)
    psr.add_argument("--tol", default=3.e-2, type=float)
    args = psr.parse_args()

    # Get rating matrix
    R, R_test = get_rating(data_dir=args.data_dir, base=args.base, test=args.test, debug=True)

    # Training
    mf = MatrixFactorization(R=R, R_test=R_test, K=args.K, a=args.a, b=args.b)
    print("training...")
    mf.train(tol=args.tol, debug=True)

    #print("The number of test data is {}.".format(np.size(R_test.nonzero()[0])))
    #print("The number of correct predictions is {}.".format(np.sum((np.round(mf.rating_matrix()) == R_test).astype(int))))

if __name__ == "__main__":
    main()
