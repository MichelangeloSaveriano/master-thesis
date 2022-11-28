import numpy as np
import math
# from tqdm.notebook import tqdm
from tqdm import tqdm
from numba import jit


def Adj(w):
    p = int((1 + math.sqrt(8 * w.size + 1)) / 2)
    idx_1, idx_2 = np.triu_indices(p, k=1)
    A = np.zeros((p, p))
    A[idx_1, idx_2] = w
    A[idx_2, idx_1] = w
    return A


def Deg(w):
    degree = Adj(w).sum(axis=0)
    return degree


def Lap(w):
    A = Adj(w)
    L = np.diag(A.sum(axis=0)) - A
    return L


def AdjStar(M):
    p = M.shape[0]
    j, l = np.triu_indices(p, k=1)
    return M[l, j] + M[j, l]
    # return w


def LapStar(M):
    p = M.shape[0]
    j, l = np.triu_indices(p, k=1)
    return M[j, j] + M[l, l] - (M[l, j] + M[j, l])


def DegStar(w):
    return LapStar(np.diag(w))


def AdjInv(A):
    idx_1, idx_2 = np.triu_indices(A.shape[0], k=1)
    w = A[idx_1, idx_2]
    return w


def LapInv(A):
    idx_1, idx_2 = np.triu_indices(A.shape[0], k=1)
    w = -A[idx_1, idx_2]
    return w


def learn_connected_graph(S, w0=None, d=1,
                          rho=1, maxiter=300,
                          reltol=1e-5, verbose=True,
                          mu=2, tau=2):
    # number of nodes
    p = S.shape[0]
    # w-initialization
    if w0 is None:
        # w = LapInv(np.linalg.inv(S))
        # A0 = Adj(w)
        # A0 = A0 / A0.sum(axis=0, keepdims=True)
        # w = AdjInv(A0)
        w = (p * (p - 1) // 2) * np.ones(p * (p - 1) // 2)
        A0 = Adj(w)
        A0 = A0 / A0.sum(axis=0, keepdims=True)
        w = AdjInv(A0)
    else:
        w = w0

    LstarS = LapStar(S)
    J = np.ones((p, p)) / p

    # Theta-initilization
    Lw = Lap(w)
    Theta = Lw
    Y = np.zeros((p, p))
    y = np.zeros(p)

    it = range(maxiter)
    if verbose:
        it = tqdm(it)

    has_converged = False
    i = 0

    for i in it:
        # update w
        LstarLw = LapStar(Lw)
        DstarDw = DegStar(np.diag(Lw))
        grad = LstarS - LapStar(Y + rho * Theta) + DegStar(y - rho * d) + rho * (LstarLw + DstarDw)
        eta = 1 / (2 * rho * (2 * p - 1))
        wi = w - eta * grad
        wi[wi < 0] = 0
        Lwi = Lap(wi)
        # update Theta
        gamma, V = np.linalg.eigh(rho * (Lwi + J) - Y)
        Thetai = V @ np.diag((gamma + np.sqrt(gamma ** 2 + 4 * rho)) / (2 * rho)) @ V.T - J
        # update Y
        R1 = Thetai - Lwi
        Y = Y + rho * R1
        # update y
        R2 = np.diag(Lwi) - d
        y = y + rho * R2
        # update rho
        s = rho * np.linalg.norm(LapStar(Theta - Thetai), 2)
        r = np.linalg.norm(R1, "fro")
        if r > mu * s:
            rho = rho * tau
        elif s > mu * r:
            rho = rho / tau
        error = np.linalg.norm(Lwi - Lw, 'fro') / np.linalg.norm(Lw, 'fro')

        # print(f'iter {i + 1}) Error: {error}, Sparcity: {(np.abs(wi) < 1e-5).mean()}, rho: {rho}')

        w = wi
        Lw = Lwi
        Theta = Thetai

        has_converged = (error < reltol) and (i > 1)
        if has_converged:
            break

    results = {'L': Lap(w),
               'A': Adj(w),
               'w': w,
               'maxiter': i,
               'rho': rho,
               'convergence': has_converged,
               }
    return results
