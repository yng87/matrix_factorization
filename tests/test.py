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
    train_data = np.genfromtxt(data_dir+base, delimiter="\t", usecols=(0,1,2))
    test_data = np.genfromtxt(data_dir+test, delimiter="\t", usecols=(0,1,2))
    item = pd.read_table(data_dir+"u.item", sep="|", header=None, encoding='latin-1')
    user = pd.read_table(data_dir+"u.user", delimiter="|", header=None)

    # Name each column
    item.columns = ["id", "title", "release date", "video release date", "url", "unknown", 
                "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
               "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi",
               "thriller", "war", "western"]
    user.columns = ["id", "age", "gender", "occupation", "zip"]

    # # Get the number of users and items
    user_size = np.size(user["id"])
    item_size = np.size(item["id"])

    return train_data, test_data, user_size, item_size

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
    R, R_test, user_size, item_size = get_rating(data_dir=args.data_dir, base=args.base, test=args.test, debug=True)

    # Training
    mf = MatrixFactorization(R=R, R_test=R_test, user_size=user_size, item_size=item_size, K=args.K, a=args.a, b=args.b)
    print("training...")
    mf.train(tol=args.tol, debug=True)

    print("The number of test data is {}.".format(R_test.shape[0]))

    icor = 0
    for r in R_test:
        iu = int(r[0])-1
        ii = int(r[1])-1
        Rhatui = mf.rating(iu=iu, ii=ii)
        if np.round(Rhatui) == r[2]:icor = icor + 1
 
    print("The number of correct predictions is {}.".format(icor))

if __name__ == "__main__":
    main()
