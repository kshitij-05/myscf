import numpy as np
import scipy

def to_pauli(X):
	n2 = X.shape[0]
	n1 = int(n2/2)
	X0 = 0.5*(X[:n1,:n1] + X[n1:,n1:])
	Xz = 0.5*(X[:n1,:n1] - X[n1:,n1:])
	Xx = 0.5*(X[:n1,n1:] + X[n1:,:n1])
	Xy = (1.0j)* 0.5*(X[:n1,n1:] - X[n1:,:n1])
	return [X0,Xz,Xx,Xy]

def from_pauli(Xp):
	n = Xp[0].shape[0]
	X = np.zeros((2*n,2*n),dtype = Xp[0].dtype)
	X[:n,:n] = Xp[0] + Xp[1]
	X[n:,n:] = Xp[0] - Xp[1]
	X[n:,:n] = Xp[2] + 1.0j*Xp[3]
	X[:n,n:] = Xp[2] - 1.0j*Xp[3]
	return X

def copy_diag(A):
	n1,n2  = A.shape
	b1 = 2*n1
	b2 = 2*n2
	B = np.zeros((b1,b2), dtype = "complex128")
	B[:n1,:n2] = A 
	B[n1:,n2:] = A
	return B

def inv_sqrt(A):
	u,s,vh = scipy.linalg.svd(A)
	s_inv = np.diag(1./np.sqrt(s))
	return np.einsum("ij,jk,kl->il",u,s_inv,vh)