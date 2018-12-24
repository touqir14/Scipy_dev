import numpy as np
from scipy.linalg.lapack import dsysv

def LDL(A):
    A = np.matrix(A)
    S = np.diag(np.diag(A))
    Sinv = np.diag(1/np.diag(A))
    D = np.matrix(S.dot(S))
    Lch = np.linalg.cholesky(A)
    L = np.matrix(Lch.dot(Sinv))
    return L, D

