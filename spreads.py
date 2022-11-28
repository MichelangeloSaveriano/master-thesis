import pandas as pd
import numpy as np
from learn_graph_kalofolias import log_degree_barrier
from graph_learning_algorithm import learn_connected_graph


def compute_identity_L():
    def L_identity(X_train):
        L = np.eye(X_train.shape[1])
        L_sqrt = L
        return L, L_sqrt

    return L_identity


def compute_corr_L(k, use_abs=True):
    def L_corr(X_train):
        p = X_train.shape[1]
        S = np.corrcoef(X_train, rowvar=False)
        S_sign = np.ones_like(S)

        if use_abs:
            S_sign = np.sign(S)
            S *= S_sign

        L = np.eye(p)
        L[np.hstack([np.arange(p).reshape((-1, 1))] * k),
          np.argsort(S)[:, -(k + 1):-1]] = -1 / k

        if use_abs:
            L *= S_sign

        L_sqrt = L
        return L, L_sqrt

    return L_corr


def compute_SGS_L(alpha=1, beta=1, laplacian_root=True,
                  adjust_laplacian=True, threshold=1e-5, normalize=True):
    def L_SGS(X_train):
        if normalize:
            X_train = X_train / (X_train ** 2).sum()
        A = log_degree_barrier(X_train.T, alpha=alpha, beta=alpha)
        A *= A > threshold
        L = np.diag(A.sum(axis=0)) - A
        L_sqrt = L

        if laplacian_root:
            lambdas, V = np.linalg.eigh(L)
            L_sqrt = V @ np.diag(np.sqrt(np.fmax(0, lambdas))) @ V.T

        if adjust_laplacian:
            A_sqrt = np.diag(np.diag(L_sqrt)) - L_sqrt
            X_train_pred = X_train @ A_sqrt
            beta = (X_train * X_train_pred).sum(axis=0) / (X_train ** 2).sum(axis=0)
            L_sqrt = (np.diag(1 / beta) - A_sqrt)
        return L, L_sqrt

    return L_SGS


def compute_LGMRF_L(use_correlations=True, use_abs=True, reltol=3e-4,
                    laplacian_root=True, adjust_laplacian=True):
    def L_LGMRF(X_train):
        S = X_train.T @ X_train
        S_sign = np.ones_like(S)
        if use_correlations:
            prec = np.diag(1 / np.sqrt(np.diag(S)))
            S = (prec @ S @ prec)
        if use_abs:
            S_sign = np.sign(S)
            S *= S_sign

        results_LGMRF = learn_connected_graph(S, verbose=False, rho=1,
                                              reltol=reltol, maxiter=1000)
        L = results_LGMRF['L']
        L_sqrt = L

        if laplacian_root:
            lambdas, V = np.linalg.eigh(L)
            L_sqrt = V @ np.diag(np.sqrt(np.fmax(0, lambdas))) @ V.T

        if use_abs:
            L *= S_sign
            L_sqrt *= S_sign

        if adjust_laplacian:
            A_sqrt = np.diag(np.diag(L_sqrt)) - L_sqrt
            X_train_pred = X_train @ A_sqrt
            beta = (X_train * X_train_pred).sum(axis=0) / (X_train ** 2).sum(axis=0)
            L_sqrt = (np.diag(1 / beta) - A_sqrt)
        return L, L_sqrt

    return L_LGMRF


def compute_spreads(X_train, X_test, method, fit_intercept=False):
    train_returns = X_train.values
    test_returns = X_test.values

    if fit_intercept:
        train_returns = np.hstack([train_returns, np.ones((train_returns.shape[0], 1))])
        test_returns = np.hstack([test_returns, np.ones((test_returns.shape[0], 1))])

    L, L_sqrt = method(train_returns)

    spreads_train = train_returns @ L_sqrt
    spreads_test = test_returns @ L_sqrt

    if fit_intercept:
        spreads_train = spreads_train[:, :-1]
        spreads_test = spreads_test[:, :-1]

    return (pd.DataFrame(spreads_train, index=X_train.index, columns=X_train.columns),
            pd.DataFrame(spreads_test, index=X_test.index, columns=X_test.columns),
            L, L_sqrt)
