import numpy as np
import random
import time
# from scipy.sparse import random as sparse_random
from scipy import stats
from rref import rref
from _matrix_compress import _build_matrix_rank_k
import sys

def _build_intersection(S, J, N, Z, t_vec):
    len_S = len(S)
    if len_S >= 2:
        S1 = S[:int(len_S/2)]
        S2 = S[int(len_S/2):]
        J1 = _build_intersection(S1, J, N, Z, t_vec)
        M = _compute_M(N.shape, N, Z, S2, J1)
        J2 = _build_intersection(S2, J.union(J1), M, Z, t_vec)
        return J1.union(J2)
    else:
        i = S[0]
        print("S[0]:",i,"1/t_vec[i]:",1 / t_vec[i], "N[i,i]:",N[i,i],"\n")
        print(N)
        if np.isclose(N[i,i], 1 / t_vec[i], rtol=0, atol=1e-05):
            return {i}
        else:
            return set()



def _compute_M(M_shape, N, Z, S2, J1):
    # M = np.zeros(M_shape)
    # if len(J1) == 0:
    #     return N
    J1_idxs = list(J1)
    try:
        N[S2, S2] = N[S2, S2] + np.linalg.multi_dot([N[S2, J1_idxs], np.linalg.inv( np.eye(len(J1_idxs)) - Z[J1_idxs, J1_idxs].dot(N[J1_idxs, J1_idxs]) ), Z[J1_idxs, J1_idxs], N[J1_idxs, S2]])
    except:
        print("S2 : ", S2, " J1_idxs : ", J1_idxs)
        print("M[S2, S2] shape: ", M[S2, S2].shape)
        print("N[S2, S2] shape: ", N[S2, S2].shape)
        print("N[S2, J1_idxs] shape :", N[S2, J1_idxs].shape)
        print("Z[J1_idxs, J1_idxs] shape : ", Z[J1_idxs, J1_idxs].shape)
        print("N[J1_idxs, J1_idxs] shape : ", N[J1_idxs, J1_idxs].shape)
        print("N[J1_idxs, S2] shape : ", N[J1_idxs, S2].shape)
        sys.exit()

    return N


def _compute_Z_inverse(Q1, Q2, t_vec):
    r, n = Q1.shape
    T = np.diag(t_vec)
    T_inv = np.diag(1 / t_vec)
    Z = np.zeros((r + n, r + n))
    Z[:r, r:], Z[r:, :r], Z[r:, r:] = Q1, Q2, T

    Z_inv = np.zeros((r + n, r + n))
    Y_inv = np.linalg.pinv(-Q1.dot(T_inv).dot(Q2))
    Z_inv[:r , :r] = Y_inv
    Z_inv[r: , :r] = -T_inv.dot(Q2).dot(Y_inv)
    Z_inv[:r , r:] = -Y_inv.dot(Q1).dot(T_inv)
    # Z_inv[r: , r:] = T_inv + T_inv.dot(Q2).dot(Y_inv).dot(Q1).dot(T_inv)
    Z_inv[r: , r:] = T_inv + np.linalg.multi_dot([T_inv, Q2, Y_inv, Q1, T_inv])

    return Z, Z_inv


def test_compute_Z_inverse():
    rows = 4
    cols = 4
    passing = True
    rank_range = [1, min(rows, cols)+1]
    # rank_range = [1,2]

    for rank in range(rank_range[0], rank_range[1]):
        Q1 = _build_matrix_rank_k(rows, cols, rank)
        Q2 = _build_matrix_rank_k(rows, cols, rank).T
        # Q1 = 10000*X1
        # Q2 = 10000*X1.T
        t_vec = np.random.uniform(low=1, high=100, size=(cols))
        Z, Z_inv = _compute_Z_inverse(Q1, Q2, t_vec)
        true_Z_inv = np.linalg.pinv(Z)
        if not np.allclose(Z_inv, true_Z_inv, atol=1e-05, rtol=0): 
            print("Did NOT pass for shape : ", [rows, cols], " rank: ", rank)
            print("Z_inv : \n", Z_inv)
            print("true Z_inv: \n", true_Z_inv)
            passing = False

    if passing:
        print("PASSED all Tests!")
    else:
        print("FAILURE!")


def matroid_Intersection(M1, M2):
    n = M1.shape[1]
    S = np.arange(len(M1))
    t_vec = np.random.uniform(low=1, high=100, size=(n))
    Z, Z_inv = _compute_Z_inverse(M1, M2.T, t_vec)
    print(Z)
    print(Z_inv) 
    print(1 / t_vec)
    J = _build_intersection(S, set(), Z_inv, Z, t_vec)
    return J

def compute_common_basis(M1, M2):
    J = matroid_Intersection(M1, M2)
    print(J)
    return J

X1 = 1000*np.array([[0.30740, 0.26861, 0.76295, 1.68923],
               [0.39346, 0.96246, 0.91850, 4.63677],
               [0.27505, 0.27456, 0.31779, 1.64835],
               [0.24657, 0.84136, 0.19348, 3.85858]])

X2 = np.array([[1.03156, 0.26861, 0.76295, 1.68923],
               [1.88096, 0.96246, 0.9185 , 4.63677],
               [0.59235, 0.27456, 0.31779, 1.64835],
               [1.03484, 0.84136, 0.19348, 3.85858]])

if __name__=="__main__":

    # test_compute_Z_inverse()
    compute_common_basis(X1, X1)
    # compute_common_basis(X2, X2)