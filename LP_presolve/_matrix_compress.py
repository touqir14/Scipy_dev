import numpy as np
import random
import time
from scipy.sparse import random as sparse_random
from scipy import stats
import rref
import scipy.linalg.interpolative as ID
from scipy.linalg import qr
# import rref_cy.rref_cy as rref

def _build_magical_graph(num_nodes_X, num_groups_Y):
    """
    Builds a magical graph. A magical graph is a bipartite graph
    generated by randomly adding edges between the nodes of two
    identical sets such that the degree of each node is at most 2.
    Additionally, one of the set (Y) needs to be partitioned into
    num_groups_Y number of partitions such that the total edge
    connections is 2 * |X| / num_groups_Y, where X is the other set.
    The nodes are denoted by integers.

    Parameters
    ----------

    num_nodes_X : (Integer)
    Refers to the total number of columns of matrix A
    which is to be compressed by _matrix_compressor

    num_groups_Y : (Integer)
    Refers to the number of columns the newly compressed
    matrix B needs to have. Matrix B is generated from _matrix_compressor
    based on the input matrix A such that
    min(rank_A, num_groups_Y) = min(rank_B, num_groups_Y).


    Returns
    ----------

    Y_edges : (2D array)
    Its dimension is [num_groups_Y, 2*num_nodes_X_new] such that num_nodes_X_new
    is either num_nodes_X or an expanded number of nodes to ensure that
    num_nodes_X_new is a multiple of num_groups_Y.
    num_nodes_X_new is the maximum number nodes that each parition of Y can
    contain. Y_edges[i,:] contains all the nodes of X that are connected to the
    nodes within partition 'i' of Y.

    num_group_edges : (1D array)
    num_group_edges[i] contains the total number of edges of the nodes within
    partition 'i' of Y.

    """

    if num_nodes_X % num_groups_Y == 0:
        size_groups_Y = int(num_nodes_X / num_groups_Y)
        num_nodes_X_new = num_nodes_X
    else:
        size_groups_Y = int(num_nodes_X / num_groups_Y) + 1
        num_nodes_X_new = num_nodes_X + \
            num_groups_Y - (num_nodes_X % num_groups_Y)

    # One side of the bipartite graph
    X1_edges = -1 * np.ones(num_nodes_X_new, dtype=int)
    # The other side of the bipartite graph
    X2_edges = -1 * np.ones(num_nodes_X_new, dtype=int)
    # At most degree of 2 for each node within the groups
    Y_edges = np.zeros((num_groups_Y, 2 * size_groups_Y), dtype=int)
    num_group_edges = np.zeros(num_groups_Y, dtype=int)

    # _build_perfect_matching1(X1_edges, X2_edges, num_nodes_X, num_nodes_X_new)
    _build_perfect_matching2(X1_edges, X2_edges)

    idx_col = 0
    for i in range(num_nodes_X_new):
        x1 = X1_edges[i]
        x2 = X2_edges[i]
        idx_row = i // size_groups_Y

        if x1 == x2:
            if x1 != -1:
                Y_edges[idx_row, idx_col] = x1
                idx_col += 1
        else:
            if x1 != -1:
                Y_edges[idx_row, idx_col] = x1
                idx_col += 1
            if x2 != -1:
                Y_edges[idx_row, idx_col] = x2
                idx_col += 1

        if i % size_groups_Y == size_groups_Y - 1:
            if idx_col <= Y_edges.shape[1] - 1:
                Y_edges[idx_row, idx_col] = -1

            num_group_edges[idx_row] = idx_col
            idx_col = 0

    # print(Y_edges)
    return Y_edges, num_group_edges


def _build_perfect_matching1(X1_edges, X2_edges, num_nodes_X, num_nodes_X_new):
    """
    Builds two perfect matching based on a fully connected graph composed on num_nodes_X_new
    number of nodes. The function randomly samples the edges within the two perfect
    matchings X1 and X2. After execution, X1_edges[i], corresponding to X1, denotes the node
    with which node i has an undirected edge. The same rule applies to X2_edges.
    With this algorithm, the union of the two matchings is not guaranteed to contain all
    the nodes in X. Also, this algorithm only permits nodes that are within the original
    X (and not the expanded X) within the matchings.

    Parameters
    ----------

    X1_edges: (1D numpy array)
    X1_edges[i] denotes the node that node i will be matched with within perfect matching X1.
    X1_edges[i] = -1 if i is not connected with any edge.

    X2_edges: (1D numpy array)
    X2_edges[i] denotes the node that node i will be matched with within perfect matching X2
    X2_edges[i] = -1 if i is not connected with any edge.

    num_nodes_X: (Integer)
    Denotes the original number of nodes of X

    num_nodes_X_new: (Integer)
    Denotes the number of nodes of expanded X (See "_build_magical_graph" function)

    Returns
    -------
    None

    """

    X1_edge_pairs = np.arange(num_nodes_X_new)
    X2_edge_pairs = np.arange(num_nodes_X_new)

    np.random.shuffle(X1_edge_pairs)
    np.random.shuffle(X2_edge_pairs)

    for i in range(int(len(X1_edge_pairs) / 2)):
        x1_first, x1_second = X1_edge_pairs[2 * i], X1_edge_pairs[2 * i + 1]
        x2_first, x2_second = X2_edge_pairs[2 * i], X2_edge_pairs[2 * i + 1]

        if x1_first < num_nodes_X and x1_second < num_nodes_X:
            X1_edges[x1_first] = x1_second
            X1_edges[x1_second] = x1_first

        if x2_first < num_nodes_X and x2_second < num_nodes_X:
            X2_edges[x2_first] = x2_second
            X2_edges[x2_second] = x2_first

    return


def _build_perfect_matching2(X1_edges, X2_edges):
    """
    Builds two perfect matchings (See function _build_perfect_matching1). This algorithm on the
    other hand produces two randomly generated perfect matchings such that their union contains
    all the nodes within expanded X.

    Parameters
    --------
    X1_edges: (1D numpy array)
    See _build_perfect_matching1 function

    X2_edges: (1D numpy array)
    See _build_perfect_matching1 function

    Returns
    -------

    None

    """

    edge_pairs = np.arange(len(X1_edges))
    np.random.shuffle(edge_pairs)
    turn1 = True

    for i in range(len(edge_pairs)):  

        if i == len(edge_pairs) - 1:
            # i is the last node
            x1, x2 = edge_pairs[0], edge_pairs[i]  
        else:
            x1, x2 = edge_pairs[i], edge_pairs[i + 1]

        if turn1:
            X1_edges[x1] = x2
            X1_edges[x2] = x1
            turn1 = False
        else:
            X2_edges[x1] = x2
            X2_edges[x2] = x1
            turn1 = True
    return


def _check_overlapping_union(Y_edges, num_group_edges, num_nodes_X):
    """
    A utility function for checking whether the edge set within Y contains all the nodes
    in X (or its expansion) and whether there exists nodes in X (or its expansion) that
    are common within the edge set of the partitions of Y.

    Parameters
    ----------

    Y_edges : (2D numpy array)
    (See the Returns in function _build_magical_graph)

    num_group_edges
    (See the Returns in function _build_magical_graph)

    num_nodes_X
    Number of nodes of original X (or its expansion).

    Returns
    -------

    None

    """

    node_set = set()
    intersection = False
    for i in range(Y_edges.shape[0]):
        nodes = set(list(Y_edges[i, 0:num_group_edges[i]]))
        if len(node_set.intersection(node_set)) != 0:
            intersection = True

        node_set = node_set.union(nodes)

    if len(node_set) == num_nodes_X:
        print("Contains all elements")
    else:
        print(
            "DOES NOT contain all elements",
            "..num of nodes:",
            len(node_set),
            "num_nodes_X:",
            num_nodes_X)
    if intersection:
        print("Has overlapping")


def _build_matrix_rank_k(row, col, k):
    """
    Builds a random matrix A (2D numpy array) of shape=(row,col) with rank k.

    Parameters
    ----------

    row: (Integer)
    Number of rows of A

    col: (Integer)
    Number of columns of A

    Returns
    -------

    A : (2D array)
    Random matrix with rank k of shape=(row,col).

    """

    # a = np.random.rand(row, col)
    a = np.random.uniform(low=-1000, high=1000, size=(row, col))
    # a = np.random.uniform(low=-10, high=10, size=(row, col))
    # a = np.random.uniform(low=-20, high=20, size=(row, col))

    # rvs = stats.uniform(loc=-10, scale=20).rvs
    # a = sparse_random(row, col, density=0.0001, data_rvs=rvs).A

    u, s, vh = np.linalg.svd(a, full_matrices=True)
    smat = np.zeros((row, col))
    smat[:k, :k] = np.diag(np.ones((k)))
    A = np.dot(u, np.dot(smat, vh))
    return A


def _hash_columns(total_columns, k):
    """
    Randomly maps new extra columns to old columns such that (total_columns + extra columns)
    is a multiple of k.

    Parameters
    --------

    total_columns : (Integer)
    Original number of columns of matrix A used within _matrix_compressor

    k : (Integer)
    The new number of collumns of the compressed matrix B (See _matrix_compressor).

    Returns
    -------

    hash_array : (1D numpy array)
    hash_array[i] = i if i is one of the original columns of A, otherwise hash_array[i] = j such
    that j is one of the original columns of A.

    """

    extra_columns = k - (total_columns % k)
    if extra_columns == k:
        extra_columns = 0
    toAdd = random.sample(range(total_columns), extra_columns)
    hash_array = np.array(list(range(total_columns)) + toAdd)
    return hash_array


def _matrix_compressor(A, K, gen_neighbours=False, info_dict=None):
    """
    For compressing input matrix A of shape=(m,n) to matrix B of shape(m,k) such that
    min(rank(A),K) = min(rank(B),K)

    Parameters
    ----------

    A : (2D numpy array)
    Matrix to compress

    K : (Integer)
    K (<n) is the number of collumns of the newly compressed matrix B

    Returns
    -------

    B : (2D numpy array)
    The compressed matrix of shape = (m,n) such that min(rank(A),K) = min(rank(B),K)

    """

    # F_set = set(A.flat) # Elements of the field (F)
    # t1 = time.time()
    # t0 = t1
    F_set = range(1, 100)
    hash_array = _hash_columns(A.shape[1], K)
    graph, num_edges = _build_magical_graph(hash_array.shape[0], K)
    # Vector of random c values
    C = np.array(random.choices(list(F_set), k=num_edges.sum()))
    B = np.zeros((A.shape[0], K))
    # print(-1, time.time() - t1)

    if gen_neighbours: 
        num_edge = num_edges[0]
        info_dict['neighbours'] = np.zeros((K, num_edge), dtype=int)

    c_idx = 0
    for i in range(B.shape[1]):
        num_edge = num_edges[i]
        if num_edge > 0:
            # t1 = time.time()
            idxs_col = graph[i, 0:num_edge]
            idxs_col = hash_array[idxs_col]
            # print(-2, time.time() - t1)
            # B[:,i] is the random linear combination of the collumns of A
            # based on C values.
            # t1 = time.time()
            B[:, i] = A[:, idxs_col].dot(C[c_idx:c_idx + num_edge])
            c_idx += num_edge
            # print(-3, time.time() - t1)

            # t1 = time.time()
            if gen_neighbours: info_dict['neighbours'][i,:] = idxs_col
            # print(-3, time.time() - t1)            
            # T = T.union(set(idxs_col.flat))
            # print("For ", i,"th collumn in B, A collumns: ", idxs_col)

    # print(len(T))
    # print(-4, time.time() - t0)
    return B


def compute_rank(A):
    """
    Computes rank of input matrix A.

    Parameters
    ----------

    A : (2D numpy array)
    Matrix for which rank will be computed

    Returns
    -------

    rank : (Integer)
    rank of A.

    """

    row, col = A.shape
    max_rank = min(A.shape)
    start = 0
    end = max_rank
    rank = 0
    mid = 0

    while True:
        if end - start == 1:
            mid = end
        else:
            mid = int((end - start) / 2) + start

        k = mid
        B = _matrix_compressor(A, k)
        C = _matrix_compressor(B.T, k)
        rank_C = np.linalg.matrix_rank(C)

        if k > rank_C:
            rank = rank_C
            break
        elif k == rank_C:
            if k == end:
                rank = max_rank
                break
            else:
                start = mid

    return rank


def auto_compress_matrix(A):
    """
    Denoting r = rank(A), it compresses A into B with r collumns such that rank(B) = r.

    Parameters
    ----------

    A : (2D numpy array)
    Input matrix of shape (m,n)

    Returns
    -------

    B : (2D numpy array)
    The compressed matrix of shape (m,r) with rank r.

    """

    estimated_rank = compute_rank(A)
    B = _matrix_compressor(A, estimated_rank)
    return B


def test_compute_rank(
        rank_range,
        A_shape,
        num_runs,
        accept_threshold=False,
        logs=False):
    """
    For testing compute_rank function. The function tests compute_rank by enumerating through
    the range of values within rank_range and num_runs and generating. For every configuration
    it generates A.

    Parameters
    ----------

    rank_range : (list)
    Composed of two elements such that 1st denotes the starting point of a range of rank values
    and the 2nd denotes the upper limit of that range. The rank corresponds to the generated 
    matrix A.

    A_shape: (list)
    A_shape = A.shape

    num_runs : (list)
    Composed of two elements similar to rank_range for denoting the range of number of runs 
    for each rank values.

    accept_threshold : (Boolean/float)
    If set to false, the function will not perform assertion checks. If given a number in [0,100],
    it will assert whether the percentage of correct outputs is at least of this value.

    logs : (Boolean)
    For indicating whether information about A rank, computed rank and run number is shown for each
    configuration

    Returns
    -------

    None
    """

    print("Running test_compute_rank ...")

    results = np.zeros(
        (num_runs, rank_range[1] - rank_range[0] + 1),
        dtype=int)

    for rank in range(rank_range[0], rank_range[1] + 1):
        for run in range(num_runs):
            A = _build_matrix_rank_k(A_shape[0], A_shape[1], rank)
            estimated_rank = compute_rank(A)

            if logs:
                print(
                    "A rank:", rank,
                    "estimated rank:", estimated_rank,
                    "num_run:", run + 1
                    )

            if rank == estimated_rank:
                results[run, rank - rank_range[0]] = 1

    success_percent = results.mean() * 100
    print("(From test_compute_rank) Success %: ", success_percent)
    if accept_threshold is not False:
        assert(success_percent >= accept_threshold)
    return


def test_matrix_compressor(
        k_range,
        rank_range,
        A_shape,
        num_runs,
        accept_threshold=False,
        logs=False):
    """
    Function similar to test_compute_rank for testing _matrix_compressor function whether min(rank(A),k) = min(rank(B),k)
    for every enumeration (See test_compute_rank). Also includes range of k values.

    Parameters
    ----------

    k_range : (list)
    Composed of two elements such that 1st denotes the starting point of a range of k values
    and the 2nd denotes the upper limit of that range. k corresponds to the number of columns of B (Compressed matrix)

    rank_range : (list)
    See test_compute_rank

    A_shape : (list)
    A_shape = A.shape

    num_runs : (list)
    See test_compute_rank.

    accept_threshold : (Boolean/float)
    See test_compute_rank.

    logs : (Boolean)
    See test_compute_rank.

    Returns
    -------

    None
    """

    print("Running test_matrix_compressor ...")

    results = np.zeros(
        (num_runs, k_range[1] - k_range[0] + 1, rank_range[1] - rank_range[0] + 1),
        dtype=int)

    for k in range(k_range[0], k_range[1] + 1):
        for rank in range(rank_range[0], rank_range[1] + 1):
            for run in range(num_runs):
                A = _build_matrix_rank_k(A_shape[0], A_shape[1], rank)
                B = _matrix_compressor(A, k)
                # print("total entries of A: ", A.shape[0]*A.shape[1])
                # print("total entries of B: ", B.shape[0]*B.shape[1])
                # print("Number of non_zeros for A:",np.count_nonzero(A))
                # print("Number of non zeros for B:",np.count_nonzero(B))
                rank_B = np.linalg.matrix_rank(B)

                if logs:
                    print(
                        "A rank: ", rank,
                        "B rank:", rank_B,
                        ". k:", k,
                        ". num_run:", run + 1
                        )

                if rank_B == min(rank, k):
                    results[run, k - k_range[0], rank - rank_range[0]] = 1

    success_percent = results.mean() * 100
    print("(From test_matrix_compressor) Success %: ", success_percent)
    if accept_threshold is not False:
        assert(success_percent >= accept_threshold)
    return


def test_auto_compress_matrix(
        rank_range,
        A_shape,
        num_runs,
        accept_threshold=False,
        logs=False):
    """
    Function similar to test_compute_rank for testing auto_compress_matrix function whether the compressed matrix B
    is of shape (m,r) with rank(B) = r, where r = rank(A) for every enumeration (See test_compute_rank).

    Parameters
    ----------

    k_range : (list)
    Composed of two elements such that 1st denotes the starting point of a range of k values
    and the 2nd denotes the upper limit of that range. k corresponds to the number of columns of B (Compressed matrix)

    rank_range : (list)
    See test_compute_rank

    A_shape : (list)
    A_shape = A.shape

    num_runs : (list)
    See test_compute_rank.

    accept_threshold : (Boolean/float)
    See test_compute_rank.

    logs : (Boolean)
    See test_compute_rank.

    Returns
    -------

    None
    """

    print("Running test_auto_compress_matrix ...")

    results = np.zeros(
        (num_runs, rank_range[1] - rank_range[0] + 1),
        dtype=int)

    for rank in range(rank_range[0], rank_range[1] + 1):
        for run in range(num_runs):
            A = _build_matrix_rank_k(A_shape[0], A_shape[1], rank)
            B = auto_compress_matrix(A)
            B_rank = np.linalg.matrix_rank(B)

            if logs:
                print(
                    "A rank:", rank,
                    ". B rank:", B_rank,
                    "B cols:", B.shape[1],
                    "num_run:", run + 1
                    )
                print("RREF of A.T :", compute_rref(A.T))
                print("RREF of B.T :", compute_rref(B.T))

            if (rank == B_rank) and (rank == B.shape[1]):
                results[run, rank - rank_range[0]] = 1

    success_percent = results.mean() * 100
    print("(From test_auto_compress_matrix) Success %: ", success_percent)
    if accept_threshold is not False:
        assert(success_percent >= accept_threshold)
    return


def test_RREF(
        rank_range,
        A_shape,
        num_runs,
        accept_threshold=False,
        logs=False):
    """
    Function similar to test_compute_rank for testing auto_compress_matrix function whether the compressed matrix B
    is of shape (m,r) with rank(B) = r, where r = rank(A) for every enumeration (See test_compute_rank).

    Parameters
    ----------

    k_range : (list)
    Composed of two elements such that 1st denotes the starting point of a range of k values
    and the 2nd denotes the upper limit of that range. k corresponds to the number of columns of B (Compressed matrix)

    rank_range : (list)
    See test_compute_rank

    A_shape : (list)
    A_shape = A.shape

    num_runs : (list)
    See test_compute_rank.

    accept_threshold : (Boolean/float)
    See test_compute_rank.

    logs : (Boolean)
    See test_compute_rank.

    Returns
    -------

    None
    """

    print("Running test_RREF ...")

    results = np.zeros(
        (num_runs, rank_range[1] - rank_range[0] + 1),
        dtype=int)

    for rank in range(rank_range[0], rank_range[1] + 1):
        for run in range(num_runs):
            A = _build_matrix_rank_k(A_shape[0], A_shape[1], rank)
            B = auto_compress_matrix(A)
            B_rank = np.linalg.matrix_rank(B)

            A_rref, A_rank = compute_rref(A.T)
            B_rref, B_rank = compute_rref(B.T)

            if logs:
                print(
                    "A rank:", rank,
                    ". B rank:", B_rank,
                    "B cols:", B.shape[1],
                    "num_run:", run + 1
                    )
                print("RREF of A.T : \n", A_rref)
                print("RREF of B.T : \n", B_rref)

            if A_rank != rank: print("Wrong RREF for A.T")
            if B_rank != B_rank : print("Wrong RREF for B.T")

            if np.allclose(A_rref[:rank,:], B_rref, atol=1e-5, rtol=0):
                results[run, rank - rank_range[0]] = 1
            else:
                print("Not same RREF for rank: ",rank)

            # if np.allclose(A_rref[:rank,:], B_rref, atol=1e-03, rtol=0) or (A_rank != rank or B_rank != B_rank):
            #     results[run, rank - rank_range[0]] = 1
            # else:
            #     print("RREF of A.T : \n", A_rref, "\n rank of A_rref:", A_rank)
            #     print("RREF of B.T : \n", B_rref, "\n rank of B_rref:", B_rank)


    success_percent = results.mean() * 100
    print("(From test_RREF) Success %: ", success_percent)
    if accept_threshold is not False:
        assert(success_percent >= accept_threshold)
    return


def compute_rref(A):

    M = rref.rref(A)
    M_array = M[0]
    # M_rank = M[1].shape[0]
    M_rank = len(M[1])
    return M_array, M_rank


def extract_independent_columns_ID(A, rank):
    if rank == 'randomized_rank':
        rank = compute_rank(A)
    elif rank == 'exact_rank':
        rank = exact_rank(A, eps=10**-5)

    col_idxs,_ = ID.interp_decomp(A, eps_or_k=rank, rand=True)

    return col_idxs[:rank]

def extract_independent_rows_ID(A, rank):
    return extract_independent_columns_ID(A.T, rank)

def exact_rank(A, eps=10**-6):
    R = qr(A, mode='r', pivoting=True, check_finite=False)
    rank = 0
    for i in range(min(R[0].shape)):
        rank += 1
        if abs(R[0][i,i]) < eps:
            return i
    return rank
    # return ID.estimate_rank(A, eps=10**-3)

def extract_independent_columns_hybrid_ID(A, c=4, k=None):

    # t1 = time.time()
    if k is None: k = compute_rank(A)
    # print(1, time.time() - t1)

    # t1 = time.time()
    A_reduced = _matrix_compressor(A.T, k).T
    # print(2, time.time() - t1)

    A_prime = A_reduced
    info_dict = {}
    T = list(range(A.shape[1]))
    i = 0

    while True:
        if c*k >= A_prime.shape[1]:
            # t1 = time.time()
            independent_columns = [T[i] for i in extract_independent_columns_ID(A_prime, k)]
            # print(3, time.time() - t1)
            return A[:, independent_columns], independent_columns

        # t1 = time.time()
        B = _matrix_compressor(A_prime, c*k, gen_neighbours=True, info_dict=info_dict)
        S = extract_independent_columns_ID(B, k)
        T = [T[i] for i in list(set(info_dict['neighbours'][S, :].flat))]
        A_prime = A_reduced[:, T]
        # print(B.shape, A_prime.shape, i, time.time() - t1)
        i +=1

def extract_independent_rows_hybrid_ID(A,c=4,k=None):
    return extract_independent_columns_hybrid_ID(A.T, c, k)

def _build_matrix_rank_k_FAST(rows, cols, rank, really_fast=False):

    A = np.random.normal(size=(rows, cols))
    if rank >= min(cols, rows):
        return A
    else:
        if really_fast:
            # A[:, rank:] = A[:, 0].reshape(A.shape[0], 1)            
            rank -= 1
            A[:, rank:] = 1.1            
        else:
            for i in range(rank, cols):
                A[:, i] = A[:, :rank].dot(np.random.normal(size=(rank)))

    return A
    # t1 = time.time()
    # computed_rank = exact_rank(A, eps=10**-5)
    # print("Target rank: ", rank, " rank of A: ", computed_rank, "time took: ", time.time() - t1)
    # print("Target rank: ", rank, " rank of A: ", np.linalg.matrix_rank(A))


def test_extract_independent_columns_hybrid_ID(
        rank_range,
        A_shape,
        num_runs,
        accept_threshold=False,
        logs=False):

    print("Running test_extract_independent_columns_hybrid_ID ...")

    results = np.zeros(
        (num_runs, rank_range[1] - rank_range[0] + 1),
        dtype=int)

    time_spent_1 = np.zeros(
        (num_runs, rank_range[1] - rank_range[0] + 1))

    time_spent_2 = np.zeros(
        (num_runs, rank_range[1] - rank_range[0] + 1))

    for rank in range(rank_range[0], rank_range[1] + 1):
        for run in range(num_runs):
            correct = False
            # A = _build_matrix_rank_k(A_shape[0], A_shape[1], rank)
            A = _build_matrix_rank_k_FAST(A_shape[0], A_shape[1], rank, really_fast=False)
           
            t1 = time.time()
            cols2 = extract_independent_columns_ID(A, rank=rank)
            time_spent_2[run, rank - rank_range[0]] = time.time() - t1

            t1 = time.time()
            _, cols1 = extract_independent_columns_hybrid_ID(A, k=rank)
            time_spent_1[run, rank - rank_range[0]] = time.time() - t1
                      
            rank_cols1 = exact_rank(A[:, cols1])
            rank_cols2 = exact_rank(A[:, cols2])
            # rank_cols2 = rank_cols1

            if (rank_cols1 == rank) and (rank_cols2 == rank):
                correct = True

            if logs:
                print("Run: ",run, " Ranks: ", rank, rank_cols1, rank_cols2)
                print("time taken for extract_independent_columns_hybrid_ID function:", time_spent_1[run, rank - rank_range[0]])
                print("time taken for extract_independent_columns_ID function:", time_spent_2[run, rank - rank_range[0]])
                if not correct:
                    print("Rank from extract_independent_columns_hybrid_ID:", rank_cols1)
                    print("Rank from extract_independent_columns_ID:", rank_cols2)

            results[run, rank - rank_range[0]] = correct

    success_percent = results.mean() * 100
    print("(From test_auto_compress_matrix) Success %: ", success_percent)
    print("Average time taken for extract_independent_columns_hybrid_ID function: ", time_spent_1.mean())
    print("Average time taken for extract_independent_columns_ID function: ", time_spent_2.mean())

    if accept_threshold is not False:
        assert(success_percent >= accept_threshold)
    return



if __name__ == "__main__":

    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    # test_matrix_compressor([3, 3], [3, 3], [6, 6], 1, logs=False)

    # test_matrix_compressor([1, 30], [1, 30], [30, 50], 10, logs=False)
    # test_compute_rank([1, 30], [30, 50], 100, logs=False)
    # test_auto_compress_matrix([1, 30], [30, 50], 5, logs=False)

    # test_auto_compress_matrix([1, 3], [3, 5], 5, logs=True)
    # test_RREF([1, 30], [30, 50], 100, logs=False)


    # k = 10
    # A = _build_matrix_rank_k(100,100, k)
    # A = _build_matrix_rank_k(10,10, 10).astype(dtype=np.double)
    
    # t1 = time.time()
    # compute_rref(A)
    # print(time.time() - t1)

    # r = np.random.rand(10000,1)
    # A = np.random.rand(10000,10000) 
    # t1 = time.time()
    # A.dot(r)
    # A.dot(r)
    # A.dot(r)
    # print(time.time() - t1)


    # t1 = time.time()
    # info_dict = {}
    # _ = _matrix_compressor(A, 4*k, gen_neighbours=True, info_dict=info_dict)
    # print(time.time() - t1)
    # print(info_dict['neighbours'])

    # t1 = time.time()    
    # _ =  auto_compress_matrix(A)
    # print(time.time() - t1)

    # t1 = time.time()    
    # _ =  compute_rank(A)
    # print(time.time() - t1)

    # t1 = time.time()    
    # _ =  np.linalg.matrix_rank(A)
    # print(time.time() - t1)


    # _build_matrix_rank_k_FAST(100,100,32)
    # _build_matrix_rank_k_FAST(100,100,2)
    # _build_matrix_rank_k_FAST(10,100,5)
    # _build_matrix_rank_k_FAST(100,10,5)
    # _build_matrix_rank_k_FAST(100000,1000,100)

    test_extract_independent_columns_hybrid_ID([10, 10], [1000, 1000], 1, logs=True)
    # test_extract_independent_columns_hybrid_ID([50, 50], [1000000, 100], 1, logs=True)