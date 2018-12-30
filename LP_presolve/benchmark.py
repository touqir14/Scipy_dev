from scipy.io import loadmat
import numpy as np
from _matrix_compress import compute_independent_columns, _extract_independent_columns, _build_matrix_rank_k_FAST
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
    cols = _extract_independent_columns(A, rank=rank)
    A_new = A[:,cols]
    rank_new = matrix_rank(A_new)
    print("remove_redundancy_ID: ", rank_new == rank, rank_new, rank)
    return rank_new


def remove_redundancy_hybrid_ID(A, rank):
    A_new, cols = compute_independent_columns(A, k=rank)
    rank_new = matrix_rank(A_new)
    print("remove_redundancy_hybrid_ID: ", rank_new == rank, rank_new, rank)
    return rank_new

def remove_redundancy_dense(A, rank):
    A_new, status = _remove_redundancy_SVD(A.T, rank)
    A_new = A_new.T
    rank_new = matrix_rank(A_new)
    print("remove_redundancy_hybrid_ID: status:", status)
    print("remove_redundancy_hybrid_ID: Ranks:", rank_new == rank, rank_new, rank)
    print("remove_redundancy_hybrid_ID: Cols:", rank_new == A_new.shape[1], A_new.shape[1])
    return rank_new


def _remove_redundancy_SVD(A, true_rank):

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
        print("A.shape: ", A.shape)
        m, n = A.shape
        s_min = s[-1] if m <= n else 0

    return A, status

def remove_redundancy_sparse(sparse_A):
    return _remove_redundancy._remove_redundancy_sparse(sparse_A, np.ones((sparse_A.shape[0],1)))


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


def measure_performance(data_saver):
    files, filenames = import_fileNames(fileName, base_location)
    all_clear, incorrect_paths = validate_fileNames(files, extension='.mat')
    
    if not all_clear:
        print("The paths below are not valid!:")
        for path in incorrect_paths:
            print(path)
        print("Exiting..")
        sys.exit()
    
    performance_dict = {}
    for file, filename in zip(files, filenames):
        performance_dict[filename] = []            
        print("***************************************")
        print("Benchmarking on", file)

        try:
            # A, sparse_A, rank = load_matrix(file)
            A, rank = load_matrix(file)
            # rank = 90
            # A = _build_matrix_rank_k_FAST(100,100,rank)
        
            estimated_rank = matrix_rank(A)
            performance_dict[filename].append(["Numpy rank and precomputed rank", estimated_rank, rank])

            t1 = time.perf_counter()
            t2 = time.process_time()
            hybrid_ID_rank = remove_redundancy_hybrid_ID(A, rank)
            t1 = time.perf_counter() - t1
            t2 = time.process_time() - t2
            print("remove_redundancy_hybrid_ID took: ", t1, t2) 
            performance_dict[filename].append(["remove_redundancy_hybrid_ID", t1, t2, hybrid_ID_rank])

            t1 = time.perf_counter()
            t2 = time.process_time()
            SVD_rank = remove_redundancy_dense(A, rank)
            t1 = time.perf_counter() - t1
            t2 = time.process_time() - t2
            print("remove_redundancy_dense took: ", t1, t2) 
            performance_dict[filename].append(["remove_redundancy_dense", t1, t2, SVD_rank])

            t1 = time.perf_counter()
            t2 = time.process_time()
            ID_rank = remove_redundancy_ID(A, rank)
            t1 = time.perf_counter() - t1
            t2 = time.process_time() - t2
            print("remove_redundancy_ID took: ", t1, t2) 
            performance_dict[filename].append(["remove_redundancy_ID", t1, t2, ID_rank])

            # t1 = time.time()
            # remove_redundancy_sparse(sparse_A)
            # t2 = time.time() - t1
            # print("remove_redundancy_sparse took: ", t2) 
            # performance_dict[filename].append(["remove_redundancy_sparse", t2])

            # data_saver(performance_dict)

        except Exception as e:
            print("An Exception Occured!", e)

    return performance_dict


def save_file(data, path):
    try:
        with open(path + ".pickle", "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print("Failed to save!", e)

def gen_data_saver(data_path):
    def data_saver(data):
        path = data_path
        save_file(data, path)

    return data_saver

if __name__ == "__main__":

    # Name of the file containing a list of matrix names (by default located in the same directory.. ** Can be changed **)
    fileName = "All_MatrixNames.txt" 
    # fileName = "All_MatrixNames.txt" 
    # path to the directory containin the matrices
    base_location = 'C:\Home\data\Singular Matrices' 
    # path of the file where the benchmarks will be saved
    save_location = 'C:\Home\Coding\Scipy_dev\LP_presolve\performances' 


    # fileName = input("Type in the MatrixName file: ")
    if len(sys.argv) == 2:    
        fileName = sys.argv[1]
    elif len(sys.argv) == 3:
        fileName = sys.argv[1]
        save_location = sys.argv[2]

    data_saver = gen_data_saver(save_location)
    benchmark = measure_performance(data_saver)
    # print(benchmark)
    # save_file(benchmark, save_location)
