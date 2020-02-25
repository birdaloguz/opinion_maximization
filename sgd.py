import pandas as pd
import numpy as np
import random

#RMSE
def rmse_score(R, Q, P):
    I = R != 0  # Indicator function which is zero for missing data
    ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
    MSE = ME ** 2
    return np.sqrt(np.sum(MSE) / np.sum(I))  # sum of squared errors


def get_rating_estimations(R, validation=False):
    n_u, n_m = R.shape
    f = 3  # Number of latent factor pairs
    lmbda = 0.50  # Regularisation strength
    gamma = 0.01  # Learning rate
    n_epochs = 6  # Number of loops through training data
    U = 3 * np.random.rand(n_u, f)  # Latent factors for users
    V = 3 * np.random.rand(n_m, f)  # Latent factors for movies

    users, items = R.nonzero()
    for epoch in range(n_epochs):
        for u, i in zip(users, items):
            e = R[u, i] - np.dot(U[u, :], V[i, :].T)  # Error for this observation
            U[u, :] += gamma * (e * V[i, :] - lmbda * U[u, :])  # Update this user's features
            V[i, :] += gamma * (e * U[u, :] - lmbda * V[i, :])  # Update this movie's features

    R_tilda = np.dot(U, V.T)

    if validation:
        val = random.randint(0, 400)
        val_v = V[val]
        V = np.delete(V, np.s_[val], axis=0)

        r_avg = np.true_divide(R.sum(0), (R != 0).sum(0)).mean()
        opinion_matrix = R_tilda - r_avg
        val_ratings = np.dot(U, val_v.T)
        return R_tilda, opinion_matrix, U, V, val, val_ratings

    r_avg = np.true_divide(R.sum(0), (R != 0).sum(0)).mean()
    opinion_matrix = R_tilda - r_avg

    return R_tilda, opinion_matrix, U, V