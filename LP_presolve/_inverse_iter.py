from scipy.optimize import minimize, approx_fprime, check_grad
import numpy as np
from _matrix_compress import _build_matrix_rank_k
import time

def inverse_iteration(A, mu, tol):
	t1 = time.time()
	v_old = np.ones((A.shape[1],1))
	v_old = v_old / np.linalg.norm(v_old)
	A_ = A - np.eye(A.shape[1])*mu
	# A_ = A
	iter_num = 0
	while True:
		iter_num += 1
		v = np.linalg.lstsq(A_, v_old)[0]
		v = v / np.linalg.norm(v)
		if np.linalg.norm(np.ndarray.flatten(v - v_old), ord=1) < tol or np.linalg.norm(np.ndarray.flatten(v + v_old), ord=1) < tol:
			break
		else:
			v_old = v 
	print("Number of iterations: ", iter_num)
	print("total time: ", time.time() - t1)
	return v

def power_iteration(A, tol):
	t1 = time.time()
	v_old = np.ones((A.shape[1],1))
	v_old = v_old / np.linalg.norm(v_old)
	# A_ = A - np.eye(A.shape[1])*mu
	# A_ = A
	iter_num = 0
	while True:
		iter_num += 1
		v = A.dot(v_old)
		# v = np.linalg.lstsq(A_, v_old)[0]
		v = v / np.linalg.norm(v)
		if np.linalg.norm(np.ndarray.flatten(v - v_old), ord=1) < tol or np.linalg.norm(np.ndarray.flatten(v + v_old), ord=1) < tol:
			break
		else:
			v_old = v 
	print("Number of iterations: ", iter_num)
	print("total time: ", time.time() - t1)
	return v



# def produce_matrix(row, col):
# 	a = np.random.uniform(low=-1000, high=1000, size=(row, col))
#     # a = np.random.uniform(low=-10, high=10, size=(row, col))
#     # a = np.random.uniform(low=-20, high=20, size=(row, col))

#     # rvs = stats.uniform(loc=-10, scale=20).rvs
#     # a = sparse_random(row, col, density=0.1, data_rvs=rvs).A

#     u, s, vh = np.linalg.svd(a, full_matrices=True)
#     smat = np.zeros((row, col))
#     smat[:k, :k] = np.diag(np.ones((k)))
#     A = np.dot(u, np.dot(smat, vh))
#     return A


def benchmark_lstsq(n, conditioner):

	A = np.random.rand(n,n-1)
	X = np.zeros((n,n))
	X[:,:n-1] = A
	b = np.ones((n,1))
	t1 = time.time()
	np.linalg.lstsq(X - np.eye(n)*conditioner, b)
	print("time taken: ", time.time() - t1)



class LBFGS_linear_system:

	def __init__(self, mu=10**-3):
		self.mu = mu
		self.return_jac = True

	def func_inv(self, x0):
		diff = self.b - self.A.dot(x0) + self.mu*x0 
		func = np.linalg.norm(diff, ord=2)**2
		return func

	def jac_inv(self, x0):
		temp = self.b - self.A.dot(x0) + self.mu*x0
		jac = 2*self.mu*temp - 2*self.A.T.dot(temp)
		return jac

	def set_Ab(self, A, b):
		self.A = A
		self.b = b.flatten()

	def func_grad_inv(self, x0):
		temp = self.b - np.dot(self.A, x0) + self.mu*x0
		func = np.linalg.norm(temp, ord=2)**2
		jac = 2*self.mu*temp - 2*np.dot(temp, self.A)
		# print("printing gradient L1 norm:", np.linalg.norm(jac.flatten(), ord=1))
		if self.return_jac:
			return func, jac.flatten()
		else:
			return func


	def solve_linear_system(self, A, b, x0):
		"""
		Approximately solves ||b - Ax||_F^2 for x
		"""
		# fun = func_inv
		# jac = jac_inv
		self.A = A
		self.b = b.flatten()
		x0 = x0.flatten()
		inv = minimize(self.func_grad_inv, x0, method='L-BFGS-B',
				 jac=True, options={'disp':True})
		return inv.x


def gen_testMat1(n, eig_range, rank):
	mat = np.zeros((n))
	mat[:rank] = np.linspace(eig_range[0], eig_range[1], rank) 
	mat += 1000
	# return np.diag(mat) + np.random.uniform(low=0.000001, high=0.001, size=(n,n))
	return np.diag(mat)

def test_LBFGS_system_solver(n, eig_range, rank, gen_mat_func, mu=10**-3):
	A = gen_mat_func(n, eig_range, rank)
	b = np.ones((n,1))
	x0 = np.zeros((n,1))
	solver = LBFGS_linear_system(mu)

	# eps = np.sqrt(np.finfo(float).eps)
	# solver.set_Ab(A, b)
	# x = np.random.rand(n)
	# _, jac = solver.func_grad_inv(x)
	# approx_jac = approx_fprime(x, solver.func_inv, eps*np.ones((n)))

	# print("abs diff: ", np.linalg.norm(jac.flatten() - approx_jac.flatten(), ord=1))

	# if np.allclose(jac, approx_jac, rtol=0, atol=10**-6):
	# 	print("Correct gradient")
	# else:
	# 	print("Incorrect gradient!")

	# print(check_grad(solver.func_inv, solver.jac_inv, x))


	t1 = time.time()
	solution1 = solver.solve_linear_system(A, b, x0)
	runtime1 = time.time() - t1

	# A_ = A - np.eye(n)*mu
	# t1 = time.time()
	# solution2 = np.linalg.lstsq(A_, b)[0]
	# runtime2 = time.time() - t1

	# print("abs diff: ", np.linalg.norm(solution1.flatten() - solution2.flatten(), ord=1))
	# if np.allclose(solution1, solution2, rtol=0, atol=10**-4):
	# 	print("Test passed!")
	# else:
	# 	print("Test failed!")
	print("runtime of LBFGS:, ", runtime1)
	# print("runtime of lstsq:, ", runtime2)

	return

def checkGradient():
	# numerical gradient checking
	# f(x + t * delta) - f(x - t * delta) / (2t)
	# should be roughly equal to inner product <g, delta>
	np.random.seed(1)
	n = 10000
	A = np.random.randn(n,n)
	c = np.random.randn(1)
	b = np.random.randn(n)
	x = np.random.randn(n)
	t = 1E-10
	solver = LBFGS_linear_system(c)
	solver.set_Ab(A, b)

	delta = np.random.randn(n)
	f1, _ = solver.func_grad_inv(x + t * delta)
	f2, _ = solver.func_grad_inv(x - t * delta)
	f, g = solver.func_grad_inv(x)
	print('approximation error',
		  np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def benchmark_svd():
	n = 10000
	A = np.random.rand(n,n)
	t1 = time.time()
	_ = np.linalg.eigh(A)
	print("time taken: ", time.time() - t1)

if __name__ == "__main__":
	# n = 1000
	# A = np.random.rand(n,n-1)
	# X = np.zeros((n,n))
	# X[:,:n-1] = A
	# X[:,-1] = A[:,0] + A[:,1]

	tol = 10**-4
	# eigvec = inverse_iteration(X, 0.0000001, tol)
	
	# benchmark_svd()

	# d,u = np.linalg.eig(X)
	# print("eigvec: \n", eigvec)
	# print("d: \n", d)
	# print("u: \n", u)
	# print(X.dot(eigvec))

	# benchmark_lstsq(10000, 10**-1)

	n = 10000
	max_eig = 100
	eig_range = [0.01,max_eig]
	rank = n-1
	gen_mat_func = gen_testMat1
	mu = 1000 + 0.001
	test_LBFGS_system_solver(n, eig_range, rank, gen_mat_func, mu)

	# checkGradient()

	# A = gen_testMat1(n, eig_range, rank)
	# A_ = max_eig*np.eye(n) - A  
	# eigvec = power_iteration(A_, tol/n)
	# print(np.linalg.norm(A.dot(eigvec).flatten(), ord=1)/n)