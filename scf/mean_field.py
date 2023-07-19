from integrals.ao_integrals import ao_integrals
from utils.maths import from_pauli, to_pauli, copy_diag, inv_sqrt
from utils.parse_inp import parse_input 
from pyscf import gto, scf, lib
import scipy
import numpy as np

class my_scf:
	def __init__(self,inp):
		self.ints = ao_integrals(inp)
		self.scf = inp.scf
		self.convergence = inp.scf_convergence
		self.ncomponent = 1
		self.s_inv = None

	def compute_density(self,f,n):
		s_inv = self.s_inv
		f_orth = np.einsum("ij,jk,kl->il",s_inv.T, f, s_inv)
		e,c_ = scipy.linalg.eigh(f_orth)
		c = np.einsum("ij,jk->ik",s_inv,c_)
		d = np.einsum("ij,kj->ik",c[:,:n],c[:,:n])
		return d

	def build_fock(self,h,d):
		return h + np.einsum("rs,pqrs->pq", d, 2.*self.ints.eri - self.ints.eri.transpose(0,2,1,3))

	def compute_energy(self,d,h,f):
		energy = np.einsum("ij,ij",h+f,d.conj().transpose())
		return energy

	def do_scf(self):
		self.ints.get_one_body()
		self.ints.get_two_body()
		ints = self.ints
		nocc = int(ints.nelec/2)
		old_energy = 0.0
		hcore = ints.compute_hcore()
		f = hcore
		self.s_inv =inv_sqrt(self.ints.s)
		for iterr in range(100):
			if ints.x2c:
				nocc = ints.nelec
			d = self.compute_density(f, nocc)
			energy = self.compute_energy(d,hcore,f)
			print(energy + ints.enuc)
			f = self.build_fock(hcore,d)
			diff = abs(energy - old_energy)
			old_energy = energy
			if diff <= self.convergence:
				break


class gmf:
	def __init__(self,inp):
		self.interface = inp.interface
		self.inp = inp

	def do_scf(self):

		if self.inp.scf["interface"]=="my_scf":
			mf = my_scf(self.inp)
			mf.do_scf()
			return mf

		elif self.inp.scf["interface"]=="pyscf":

			if self.inp.scf["type"]=="rhf":
				mf = scf.RHF(self.inp.molecule())
				mf.kernel()
				return mf
			
			if self.inp.scf["type"]=="uhf":
				mf = scf.UHF(self.inp.molecule())
				mf.kernel()
				return mf

			elif self.inp.scf["type"]=="sfx2c":
				mf = scf.GHF(self.inp.molecule()).sfx2c1e()
				mf.kernel()
				return mf

			elif self.inp.scf["type"]=="x2c":
				mf = scf.X2C(self.inp.molecule())
				mf.kernel()
				return mf














