import pandas as pd
import numpy as np
import random
from scipy.sparse.linalg import svds

def prediction(P,Q):
    return np.dot(P.T,Q)

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def train_test_split(ratings):
    validation = np.zeros(ratings.shape)
    train = ratings.copy()  # don't do train=ratings, other wise, ratings becomes empty

    for user in np.arange(ratings.shape[0]):
        if len(ratings[user, :].nonzero()[
                   0]) >= 35:  # 35 seems to be best, it depends on sparsity of your user-item matrix
            val_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                           size=15,  # tweak this, 15 seems to be optimal
                                           replace=False)
            train[user, val_ratings] = 0
            validation[user, val_ratings] = ratings[user, val_ratings]
    return train, validation

def get_rating_estimations(A):
    train_errors = []
    val_errors = []

    train, val = train_test_split(A)

    # Only consider items with ratings
    users, items = A.nonzero()

    lmbda = 0.4 # Regularization parameter
    k = 3 #tweak this parameter
    m, n = A.shape  # Number of users and items
    n_epochs = 100  # Number of epochs
    alpha=0.01  # Learning rate

    U = 3 * np.random.rand(k,m) # Latent user feature matrix
    V = 3 * np.random.rand(k,n) # Latent movie feature matrix
    for epoch in range(n_epochs):
        for u, i in zip(users, items):
            e = A[u, i] - prediction(U[:, u], V[:, i])  # Calculate error for gradient update
            U[:, u] += alpha * (e * V[:, i] - lmbda * U[:, u])  # Update latent user feature matrix
            V[:, i] += alpha * (e * U[:, u] - lmbda * V[:, i])  # Update latent item feature matrix

        train_rmse = rmse(prediction(U, V), A)
        val_rmse = rmse(prediction(U, V), val)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

    R_tilda = np.dot(U.T, V)
    r_avg = np.true_divide(R_tilda.sum(0), (R_tilda != 0).sum(0)).mean()
    opinion_matrix = R_tilda - r_avg

    return R_tilda, opinion_matrix, U, V, r_avg