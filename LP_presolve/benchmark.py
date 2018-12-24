from scipy.io import loadmat
import numpy as np
from _matrix_compress import compute_independent_columns 
import os.path 
from scipy.optimize import _remove_redundancy
import time
from numpy.linalg import matrix_rank
import scipy.sparse
import pickle
import copy
from pathlib import Path
import sys


def load_matrix(fileName):
    M_dict = loadmat(fileName)
    rank = M_dict['numrank'][0,0]
    if type(M_dict['A']) == np.ndarray:
        A = M_dict['A']
        sparse_A = scipy.sparse.csr_matrix(A)
        return A, sparse_A,  rank
    else:
        sparse_A = M_dict['A']
        A = sparse_A.toarray() 
        return A, sparse_A, rank


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


# def remove_redundancy_ID(A, rank):
#     return _extract_independent_columns(A, rank)

def remove_redundancy_hybrid_ID(A, rank):
    return compute_independent_columns(A, k=rank)

def remove_redundancy_SVD(A):
    return _remove_redundancy._remove_redundancy(A, np.ones((A.shape[0],1)))

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


def measure_performance():
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
            A, sparse_A, rank = load_matrix(file)

            t1 = time.time()
            remove_redundancy_hybrid_ID(A, rank)
            t2 = time.time() - t1
            print("remove_redundancy_hybrid_ID took: ", t2) 
            performance_dict[filename].append(["remove_redundancy_hybrid_ID", t2])

            t1 = time.time()
            remove_redundancy_SVD(A)
            t2 = time.time() - t1
            print("remove_redundancy_SVD took: ", t2) 
            performance_dict[filename].append(["remove_redundancy_SVD", t2])

            t1 = time.time()
            remove_redundancy_sparse(sparse_A)
            t2 = time.time() - t1
            print("remove_redundancy_sparse took: ", t2) 
            performance_dict[filename].append(["remove_redundancy_sparse", t2])

        except Exception as e:
            print("An Exception Occured!", e)

    return performance_dict

def save_file(data, path):
    try:
        with open(path + ".pickle", "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print("Failed to save!", e)


if __name__ == "__main__":

    # Name of the file containing a list of matrix names (by default located in the same directory.. ** Can be changed **)
    fileName = "MatrixNames.txt" 
    # path to the directory containin the matrices
    base_location = 'C:\Home\data\Singular Matrices' 
    # path of the file where the benchmarks will be saved
    save_location = 'C:\Home\Coding\Scipy_dev\LP_presolve\performances' 


    # fileName = input("Type in the MatrixName file: ")
    benchmark = measure_performance()
    # print(benchmark)
    save_file(benchmark, save_location)
