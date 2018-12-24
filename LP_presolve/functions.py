"""
Sample code automatically generated on 2018-11-13 18:31:58

by www.matrixcalculus.org

from input

d/dx norm2(b - A*x + c*x)^2 = (2*c*(b-A*x+c*x))'-2*(b-A*x+c*x)'*A

where

A is a matrix
b is a vector
c is a scalar
x is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(A, b, c, x):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(b, np.ndarray)
    dim = b.shape
    assert len(dim) == 1
    b_rows = dim[0]
    if isinstance(c, np.ndarray):
        dim = c.shape
        assert dim == (1, )
    assert isinstance(x, np.ndarray)
    dim = x.shape
    assert len(dim) == 1
    x_rows = dim[0]
    assert A_cols == b_rows == x_rows == A_rows

    t_0 = ((b - np.dot(A, x)) + (c * x))
    functionValue = (np.linalg.norm(t_0) ** 2)
    gradient = (((2 * c) * t_0) - (2 * np.dot(t_0, A)))

    return functionValue, gradient

n=10000

def checkGradient(A, b, c, x):
    global n
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-10
    delta = np.random.randn(n)
    f1, _ = fAndG(A, b, c, x + t * delta)
    f2, _ = fAndG(A, b, c, x - t * delta)
    f, g = fAndG(A, b, c, x)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    global n
    np.random.seed(1)
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    c = np.random.randn(1)
    x = np.random.randn(n)
    print(n)
    return A, b, c, x

if __name__ == '__main__':
    A, b, c, x = generateRandomData()
    functionValue, gradient = fAndG(A, b, c, x)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, b, c, x)
