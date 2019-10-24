# coding=utf-8
"""Dimensionality reduction Principal Component Analysis algorithm implementation using Numpy."""

import numpy as np


def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset

    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the
        mean and standard deviation respectively.
    """
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std == 0] = 1.
    Xbar = (X - mu) / std_filled
    return Xbar, mu, std


def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    return np.linalg.eig(S)


def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    return B @ B.T


def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    S = projection_matrix(X) / len(X)
    lam, eig_v = eig(S)
    s_ids = np.argsort(-lam)
    lam = lam[s_ids]
    eig_v = eig_v[:, s_ids]
    B = eig_v[:, :num_components]
    P = projection_matrix(B)
    # your solution should take advantage of the functions you have implemented above.
    return P @ X  # <-- EDIT THIS to return the reconstruction of X
