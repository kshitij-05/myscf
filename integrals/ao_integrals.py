import numpy as np
from pyscf import gto

def w_transform(w,k,p_inv):
	w_ = np.einsum("ij,jk,kl->il",k.conj().transpose(),w,k)
	w_ = np.einsum("ij,jk,kl->il",p_inv,w_,p_inv)
	return w_


class ao_integrals:
	def __init__(self, inp):
		self.mol = inp.molecule()
		self.inp = inp
		self.s = None
		self.t = None
		self.v = None
		self.eri = None
		self.nelec = self.mol.nelectron
		self.nao = 0
		self.enuc = self.mol.energy_nuc()
		self.x2c = inp.x2c

	def get_one_body(self):
		if self.x2c:
			self.s = self.mol.intor("int1e_ovlp_spinor")
			self.t = self.mol.intor("int1e_kin_spinor")
			self.v = self.mol.intor("int1e_nuc_spinor")
			self.nao = self.s.shape[0]
		
		else:	
			self.s = self.mol.intor("int1e_ovlp")
			self.t = self.mol.intor("int1e_kin")
			self.v = self.mol.intor("int1e_nuc")
			self.nao = self.s.shape[0]

	def get_two_body(self):
		if self.x2c:
			self.eri = self.mol.intor("int2e")
		else:
			self.eri = self.mol.intor("int2e_spinor")
		return self.eri

	def compute_hcore(self):
		if not self.x2c:
			return self.v + self.t

