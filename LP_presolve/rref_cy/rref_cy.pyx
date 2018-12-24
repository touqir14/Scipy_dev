"""
Source : https://gist.github.com/sgsfak/77a1c08ac8a9b0af77393b24e44c9547 
"""
#cython: infer_types=True, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
def rref(double[:, :] A, double tol = 1e-8, int debug = 0):
  cdef int rows = A.shape[0]
  cdef int cols = A.shape[1]
  cdef int r = 0
  cdef int[:,:] pivots_pos = np.zeros((min(A.shape), 2), dtype=np.intc)
  cdef int[:] row_exchanges = np.arange(rows, dtype=np.intc)
  cdef int pivots_pos_idx = 0
  cdef int c, i, pivot, temp_int
  cdef double m, temp_double, max_val
  cdef double[:] temp_col = np.zeros((cols), dtype=np.double)
  cdef double[:, :] matrix_view

  for c in range(cols):
    if debug: print("Now at row", r, "and col", c, "with matrix:"); print(A)

    ## Find the pivot row:
    # pivot = np.argmax (np.abs (A[r:rows,c])) + r
    max_val = 0; pivot = 0; temp_double = 0
    for i in range(r, rows):
      temp_double = A[i, c]
      if temp_double < 0: temp_double = -1 * temp_double
      if temp_double > max_val: 
        max_val = temp_double
        pivot = i

    # m = np.abs(A[pivot, c])
    m = A[pivot, c]
    if m < 0: m = -1 * m 

    if debug: print("Found pivot", m, "in row", pivot)
    if m <= tol:
      ## Skip column c, making sure the approximately zero terms are
      ## actually zero.
      A[r:rows, c] = np.zeros(rows-r, dtype=np.double)
      if debug: print("All elements at and below (", r, ",", c, ") are zero.. moving on..")
    else:
      ## keep track of bound variables
      pivots_pos[pivots_pos_idx][0] = r
      pivots_pos[pivots_pos_idx][1] = c
      pivots_pos_idx += 1

      if pivot != r:
        ## Swap current row and pivot row
        # A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
        temp_col = A[pivot, c:cols]
        A[pivot, c:cols] = A[r, c:cols]
        A[r, c:cols] = temp_col

        # row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]
        temp_int = row_exchanges[pivot]
        row_exchanges[pivot] = row_exchanges[r]
        row_exchanges[r] = temp_int
        
        if debug: print("Swap row", r, "with row", pivot, "Now:"); print(A)

      ## Normalize pivot row
      for i in range(c,cols): A[r, i] = A[r, i] / A[r, c]
  
      ## Eliminate the current column
      v = A[r, c:cols]
      ## Above (before row r):
      if r > 0:
        # ridx_above = np.arange(r)
        # A[:r, c:cols] = np.subtract(A[:r, c:cols], temp_matrix)
        matrix_view = np.subtract(A[:r, c:cols], np.outer(v, A[:r, c]).T)
        A[:r, c:cols] = matrix_view

        if debug: print("Elimination above performed:"); print(A)
      ## Below (after row r):
      if r < rows-1:
        # ridx_below = np.arange(r+1,rows)
        # A[r+1:rows, c:cols] = np.subtract(A[r+1:rows, c:cols], temp_matrix)
        matrix_view = np.subtract(A[r+1:rows, c:cols], np.outer(v, A[r+1:rows, c]).T)
        A[r+1:rows, c:cols] = matrix_view
        if debug: print("Elimination below performed:"); print(A)
      r += 1
    ## Check if done
    if r == rows:
      break
  return (A, pivots_pos, row_exchanges)


# cdef inline void matrix_oper(double[:, :] A, double[:, :] B, double[:, :] C, char oper_type):
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
def matrix_oper(double[:, :] A, double[:, :] B, double[:, :] C, char oper_type):
  
  # cdef int rows = len(B.shape[0])
  # cdef int cols = len(col)

  cdef int i, j
  # B.shape[0] = range(B.shape[0])
  # B.shape[1] = range(B.shape[1])

  if oper_type == '+':
    for i in B.shape[0]:
      for j in B.shape[1]:
        A[i, j] = B[i, j] + C[i, j]

  elif oper_type == '-':
    for i in B.shape[0]:
      for j in B.shape[1]:
        A[i, j] = B[i, j] - C[i, j]

  elif oper_type == '*':
    for i in B.shape[0]:
      for j in B.shape[1]:
        A[i, j] = B[i, j] * C[i, j]
  
  elif oper_type == '/':
    for i in B.shape[0]:
      for j in B.shape[1]:
        A[i, j] = B[i, j] / C[i, j]


# cdef inline void outer_product(double[:, :] A, double[:] B, double[:] C):
@cython.boundscheck(False)
def outer_product(np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=1] B, np.ndarray[np.float64_t, ndim=1] C, int times):
  
  # cdef int rows = len(B.shape[0])
  # cdef int cols = len(col)

  cdef int i, j
  cdef int rows = B.shape[0]
  cdef int cols = B.shape[0]

  for t in range(times):
    for i in range(rows):
      for j in range(cols):
        A[i, j] = B[i] * C[j]

  # if oper_type == '+':
  #   for i in B.shape[0]:
  #     for j in B.shape[1]:
  #       A[i, j] = B[i, j] + C[i, j]

  # elif oper_type == '-':
  #   for i in B.shape[0]:
  #     for j in B.shape[1]:
  #       A[i, j] = B[i, j] - C[i, j]

  # elif oper_type == '*':
  #   for i in B.shape[0]:
  #     for j in B.shape[1]:
  #       A[i, j] = B[i, j] * C[i, j]
  
  # elif oper_type == '/':
  #   for i in B.shape[0]:
  #     for j in B.shape[1]:
  #       A[i, j] = B[i, j] / C[i, j]

