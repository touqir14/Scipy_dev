from scipy.io import loadmat
import numpy as np
from _matrix_compress import extract_independent_columns_hybrid_ID, extract_independent_columns_ID, _build_matrix_rank_k_FAST
import os.path 
from scipy.optimize import _remove_redundancy
import time
from numpy.linalg import matrix_rank
import scipy.sparse
import pickle
import copy
from pathlib import Path
import sys
import scipy.linalg


def load_matrix(fileName):
    M_dict = loadmat(fileName)
    rank = M_dict['numrank'][0,0]
    if type(M_dict['A']) == np.ndarray:
        A = M_dict['A']
        # sparse_A = scipy.sparse.csr_matrix(A)
        # return A, sparse_A,  rank
        return A, rank
    else:
        sparse_A = M_dict['A']
        A = sparse_A.toarray() 
        # return A, sparse_A, rank
        return A, rank


# def load_matrix_2(fileName, condMultiplier):
#     M_dict = loadmat(fileName)
#     rank = M_dict['numrank'][0,0]
#     if type(M_dict['A']) != np.ndarray: 
#         A = M_dict['A'].toarray()
#     else:
#         A = M_dict['A']
#     # print(A)
#     dim_min = min(A.shape)
#     A = A[:dim_min, :dim_min] + condMultiplier*np.eye(dim_min)
#     print("*", fileName, "*, condition number: ", np.linalg.cond(A))
#     return A, np.linalg.matrix_rank(A)


def remove_redundancy_ID(A, rank):
    cols = extract_independent_columns_ID(A, rank=rank)
    A_new = A[:,cols]

    # rows = extract_independent_rows_ID(A, rank=rank)
    # A_new = A[:,rows]

    rank_new = matrix_rank(A_new)
    print("remove_redundancy_ID: ", rank_new == rank, rank_new, rank)
    
    results = {}
    results['rank'] = rank_new
    return results


def remove_redundancy_hybrid_ID(A, rank):
    A_new, cols = extract_independent_columns_hybrid_ID(A, k=rank)
    # A_new, rows = extract_independent_columns_hybrid_ID(A, k=rank)

    rank_new = matrix_rank(A_new)
    print("remove_redundancy_hybrid_ID: ", rank_new == rank, rank_new, rank)

    results = {}
    results['rank'] = rank_new
    return results

def remove_redundancy_SVD(A, rank):
    A_new, status = _remove_redundancy_SVD(A, rank)
    rank_new = matrix_rank(A_new)
    print("remove_redundancy_SVD: status:", status)
    print("remove_redundancy_SVD: Ranks:", rank_new == rank, rank_new, rank)
    print("remove_redundancy_SVD: Cols:", rank_new == A_new.shape[1], A_new.shape[1])

    results = {}
    results['rank'] = rank_new
    return results

def remove_redundancy_dense(A, rank):
    A_new, status = _remove_redundancy_dense(A)
    rank_new = matrix_rank(A_new)
    print("remove_redundancy_dense: status:", status)
    print("remove_redundancy_dense: Ranks:", rank_new == rank, rank_new, rank)
    print("remove_redundancy_dense: Cols:", rank_new == A_new.shape[1], A_new.shape[1])

    results = {}
    results['rank'] = rank_new
    return results


def _remove_redundancy_SVD(A, true_rank):

    A = A.T
    U, s, Vh = np.linalg.svd(A)
    eps = np.finfo(float).eps
    tol = s.max() * max(A.shape) * eps

    m, n = A.shape
    s_min = s[-1] if m <= n else 0
    status = 1

    while abs(s_min) < tol:
        v = U[:, -1]  
        i_remove = np.argmax(v)
        A = np.delete(A, i_remove, axis=0)

        if min(A.shape) < true_rank or np.count_nonzero(s > tol) < true_rank:
            status = 0
            break

        U, s, Vh = np.linalg.svd(A)
        # print("A.shape: ", A.shape)
        m, n = A.shape
        s_min = s[-1] if m <= n else 0

    # return A, status
    return A.T, status

def remove_redundancy_sparse(sparse_A):
    return _remove_redundancy._remove_redundancy_sparse(sparse_A, np.ones((sparse_A.shape[0],1)))

def _row_count(A):
    """
    Counts the number of nonzeros in each row of input array A.
    Nonzeros are defined as any element with absolute value greater than
    tol = 1e-13. This value should probably be an input to the function.
    Parameters
    ----------
    A : 2-D array
        An array representing a matrix
    Returns
    -------
    rowcount : 1-D array
        Number of nonzeros in each row of A
    """
    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1)).flatten()

def _remove_zero_rows(A):
    """
    Eliminates trivial equations from system of equations defined by Ax = b
   and identifies trivial infeasibilities
    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the removal operation
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.
    """
    status = 0
    message = ""
    i_zero = _row_count(A) == 0
    A = A[np.logical_not(i_zero), :]
    return A, status, message

def bg_update_dense(plu, perm_r, v, j):
    LU, p = plu

    u = scipy.linalg.solve_triangular(LU, v[perm_r], lower=True,
                                      unit_diagonal=True)
    LU[:j+1, j] = u[:j+1]
    l = u[j+1:]
    piv = LU[j, j]
    LU[j+1:, j] += (l/piv)
    return LU, p

def _remove_redundancy_dense(A):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.
    Parameters
    ----------
    A : 2-D sparse matrix
        An matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    Returns
    ----------
    A : 2-D sparse matrix
        A matrix representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.
    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    """

    A = A.T

    tolapiv = 1e-8
    status = 0
    message = ""

    A, status, message = _remove_zero_rows(A)

    if status != 0:
        return A, status, message

    m, n = A.shape

    v = list(range(m))      # Artificial column indices.
    b = list(v)             # Basis column indices.
    # This is better as a list than a set because column order of basis matrix
    # needs to be consistent.
    k = set(range(m, m+n))  # Structural column indices.
    d = []                  # Indices of dependent rows
    lu = None
    perm_r = None

    A_orig = A
    A = np.hstack((np.eye(m), A))
    e = np.zeros(m)

    # Implements basic algorithm from [2]
    # Uses some of the suggested improvements (removing zero rows and
    # Bartels-Golub update idea).
    # Removing column singletons would be easy, but it is not as important
    # because the procedure is performed only on the equality constraint
    # matrix from the original problem - not on the canonical form matrix,
    # which would have many more column singletons due to slack variables
    # from the inequality constraints.
    # The thoughts on "crashing" the initial basis sound useful, but the
    # description of the procedure seems to assume a lot of familiarity with
    # the subject; it is not very explicit. I already went through enough
    # trouble getting the basic algorithm working, so I was not interested in
    # trying to decipher this, too. (Overall, the paper is fraught with
    # mistakes and ambiguities - which is strange, because the rest of
    # Andersen's papers are quite good.)

    B = A[:, b]
    for i in v:

        e[i] = 1
        if i > 0:
            e[i-1] = 0

        try:  # fails for i==0 and any time it gets ill-conditioned
            j = b[i-1]
            lu = bg_update_dense(lu, perm_r, A[:, j], i-1)
        except Exception:
            lu = scipy.linalg.lu_factor(B)
            LU, p = lu
            perm_r = list(range(m))
            for i1, i2 in enumerate(p):
                perm_r[i1], perm_r[i2] = perm_r[i2], perm_r[i1]

        pi = scipy.linalg.lu_solve(lu, e, trans=1)

        # not efficient, but this is not the time sink...
        js = np.array(list(k-set(b)))
        batch = 50
        dependent = True

        # This is a tiny bit faster than looping over columns indivually,
        # like for j in js: if abs(A[:,j].transpose().dot(pi)) > tolapiv:
        for j_index in range(0, len(js), batch):
            j_indices = js[np.arange(j_index, min(j_index+batch, len(js)))]

            c = abs(A[:, j_indices].transpose().dot(pi))
            if (c > tolapiv).any():
                j = js[j_index + np.argmax(c)]  # very independent column
                B[:, i] = A[:, j]
                b[i] = j
                dependent = False
                break
        if dependent:
            d.append(i)

    keep = set(range(m))
    keep = list(keep - set(d))

    # return A_orig[keep, :], status
    return A_orig[keep, :].T, status


def import_fileNames(fileName, base_location):
    with open(fileName, "r") as f:
        fileText = f.read()

    fileList = fileText.split('\n')
    fileList_original = copy.copy(fileList)

    for i in range(len(fileList)):
        if fileList[i] == '':
            del fileList[i], fileList_original[i]
            continue
        fileList[i] = os.path.join(base_location, fileList[i])

    return fileList, fileList_original

def validate_fileNames(fileList, extension=False):
    incorrect_paths = []
    all_clear = True
    for file in fileList:
        path = Path(file + extension)
        if not path.is_file(): 
            incorrect_paths.append(str(path))
            all_clear = False

    return all_clear, incorrect_paths


def measure_performance(data_saver, file_Name, base_location, save_path):
    files, filenames = import_fileNames(file_Name, base_location)
    all_clear, incorrect_paths = validate_fileNames(files, extension='.mat')
    # func_names = set(["remove_redundancy_hybrid_ID", "remove_redundancy_SVD", "remove_redundancy_ID", "remove_redundancy_dense", "Numpy rank and precomputed rank"])
    func_names = set(["remove_redundancy_hybrid_ID", "remove_redundancy_ID", "remove_redundancy_dense", "Numpy rank and precomputed rank"])
    to_compute, performance_dict, path_dict = toWorkOn(save_path, files, filenames, func_names)
#    if not all_clear:
#        print("The paths below are not valid!:")
#        for path in incorrect_paths:
#            print(path)
#        print("Exiting..")
#        sys.exit()
    
#    print(files)
#    return

    for filename in to_compute:
        print("***************************************")
        print("Benchmarking on", filename)

        try:
            # A, sparse_A, rank = load_matrix(file)
            A, rank = load_matrix(path_dict[filename])

        except Exception as e:            
            print("Failed to Load matrix!", e)
            continue

        if "Numpy rank and precomputed rank" in to_compute[filename]: 
            returns, success, _, _ = benchmark_functions(matrix_rank, A)    
            estimated_rank = returns
            print("Numpy rank and precomputed rank:", estimated_rank, rank)
            performance_dict[filename].append(["Numpy rank and precomputed rank", estimated_rank, rank])
            data_saver(performance_dict)

        if "remove_redundancy_hybrid_ID" in to_compute[filename]: 
            returns, success, dt_perf_counter, dt_process_time = benchmark_functions(remove_redundancy_hybrid_ID, A=A, rank=rank)
            print("remove_redundancy_hybrid_ID took: ", dt_perf_counter, dt_process_time)
            if type(returns) == str:
                performance_dict[filename].append(["remove_redundancy_hybrid_ID", "FAILED"])
            else:
                performance_dict[filename].append(["remove_redundancy_hybrid_ID", dt_perf_counter, dt_process_time, returns['rank']])
            data_saver(performance_dict)

        if "remove_redundancy_SVD" in to_compute[filename]: 
            returns, success, dt_perf_counter, dt_process_time = benchmark_functions(remove_redundancy_SVD, A=A, rank=rank)
            print("remove_redundancy_SVD took: ", dt_perf_counter, dt_process_time) 
            if type(returns) == str:
                performance_dict[filename].append(["remove_redundancy_SVD", "FAILED"])
            else:
                performance_dict[filename].append(["remove_redundancy_SVD", dt_perf_counter, dt_process_time, returns['rank']])
            data_saver(performance_dict)

        if "remove_redundancy_ID" in to_compute[filename]: 
            returns, success, dt_perf_counter, dt_process_time = benchmark_functions(remove_redundancy_ID, A=A, rank=rank)
            print("remove_redundancy_ID took: ", dt_perf_counter, dt_process_time)
            if type(returns) == str:
                performance_dict[filename].append(["remove_redundancy_ID", "FAILED"])
            else:
                performance_dict[filename].append(["remove_redundancy_ID", dt_perf_counter, dt_process_time, returns['rank']])
            data_saver(performance_dict)

        if "remove_redundancy_dense" in to_compute[filename]:
            returns, success, dt_perf_counter, dt_process_time = benchmark_functions(remove_redundancy_dense, A=A, rank=rank)
            print("remove_redundancy_dense took: ", dt_perf_counter, dt_process_time)
            if type(returns) == str:
                performance_dict[filename].append(["remove_redundancy_dense", "FAILED"])
            else:
                performance_dict[filename].append(["remove_redundancy_dense", dt_perf_counter, dt_process_time, returns['rank']])
            data_saver(performance_dict)

    return performance_dict


def save_file(data, path, extension_len):
    try:
        with open(path[:-(extension_len+1)] + ".pickle", "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print("Failed to save!", e)

def gen_data_saver(data_path):
    def data_saver(data):
        path = data_path
        extension_len = 3
        save_file(data, path, extension_len)

    return data_saver

def load_pickle(path):
    try:
        with open(path + ".pickle", "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        print("Failed to load!", e)

def toWorkOn(path, matrix_paths, matrix_names, func_names):
    data = load_pickle(path[:-4])
    to_compute = {}
    path_dict = {}

    if data is None:
        performance_dict = {}
        for name, loc in zip(matrix_names, matrix_paths):     
            to_compute[name] = func_names
            path_dict[name] = loc
            performance_dict[name] = []

    else:
        performance_dict = data
        for name, loc in zip(matrix_names, matrix_paths):     
            if name not in data:
                to_compute[name] = func_names
                path_dict[name] = loc
                performance_dict[name] = []
            else:
                done = set()
                for elem in data[name]:
                    if elem[1] != "FAILED":
                        done.add(elem[0])
                remaining = func_names.difference(done)
                if len(remaining) > 0:
                    to_compute[name] = remaining 
                    path_dict[name] = loc
                
    return to_compute, performance_dict, path_dict                  


def benchmark_functions(func, *args, **kwargs):
    """

    Parameters
    ----------

    func: a function
    Any redundancy removal function for benchmarking on matrix A

    **kwargs:
    Input to func

    Returns
    -------
    returns: dictionary
    Should contain at least the computed rank

    dt: float
    time taken for func to run on matrix A

    success: boolean
    Will be true if no exceptions were raised.

    """

    success = True
    t0 = time.perf_counter()
    t1 = time.process_time()

    try:
        returns = func(*args, **kwargs)
    except Exception as e:
        success = False
        returns = str(e)

    dt_perf_counter = time.perf_counter() - t0
    dt_process_time = time.process_time() - t1

    return returns, success, dt_perf_counter, dt_process_time

if __name__ == "__main__":

    # path to the directory containin the matrices
    base_location = 'C:\Home\data\Singular Matrices' 
    # path of the file where the benchmarks will be saved
    save_location = 'C:\Home\Coding\Scipy_dev\LP_presolve\performance' 


    fileName = input("Type in the MatrixName file: ")

    save_path = os.path.join(save_location, fileName)
    data_saver = gen_data_saver(save_path)
    benchmark = measure_performance(data_saver, fileName, base_location, save_path)
    # print(benchmark)
    # save_file(benchmark, save_location)
