from pyscf import lib
import numpy as np
from integrals.lcao_integrals import lcao_integrals
from cc import intermidiates as imd

#einsum = np.einsum
einsum = lib.einsum


class ccsd:
	def __init__(self,mf,inp):
		self.ref = inp.scf["type"]
		self.lcao_ints = lcao_integrals(mf, inp)
		self.t1 = np.zeros_like(self.lcao_ints.get1b("<a|f|i>"))
		self.t2 = np.zeros_like(self.lcao_ints.get2b("<ab||ij>"))


	def energy(self):
		e = einsum('ia,ai',self.lcao_ints.get1b("<i|f|a>"), self.t1)
		e += 0.25*einsum('abij,abij', self.t2, self.lcao_ints.get2b("<ab||ij>").conj(), optimize = "greedy")
		e += 0.5 *einsum('ai,bj,abij', self.t1, self.t1,self.lcao_ints.get2b("<ab||ij>").conj(),optimize = "greedy")
		if abs(e.imag) > 1e-4:
			print('Non-zero imaginary part found in GCCSD energy %s', e)
		return e.real

	
	def kernel(self):

		print("\n Begining CCSD \n")
		
		old_energy = 0
		for i in range(0,80):
			self.t1, self.t2 = self.update_amps()
			eccsd = self.energy()
			delta_e = np.abs(old_energy-eccsd)
			print(f"iter: {i} ecc = {eccsd} \t deltae = {delta_e} ")
			if delta_e <= 1e-9:
				break
			old_energy = eccsd

	def update_amps(self):

		tau = imd.make_tau(self.t2, self.t1, self.t1)

		Fvv = imd.cc_Fvv(self.t1, self.t2, self.lcao_ints)
		Foo = imd.cc_Foo(self.t1, self.t2, self.lcao_ints)
		Fov = imd.cc_Fov(self.t1, self.t2, self.lcao_ints)
		Woooo = imd.cc_Woooo(self.t1, self.t2, self.lcao_ints)
		Wvvvv = imd.cc_Wvvvv(self.t1, self.t2, self.lcao_ints)
		Wovvo = imd.cc_Wovvo(self.t1, self.t2, self.lcao_ints)
		eia, eijab = self.lcao_ints.get_denom2()

		# T1 equation
		t1new = self.lcao_ints.get1b("<a|f|i>")
		t1new +=  einsum('ae,ei->ai', Fvv ,self.t1)
		t1new -=  einsum('am,mi->ai',self.t1,Foo)
		t1new +=  einsum('me,aeim->ai', Fov,self.t2)
		t1new -= einsum('em,amei->ai', self.t1, self.lcao_ints.get2b("<ai||bj>").conj())
		t1new += 0.5*einsum('efim,efam->ai', self.t2, self.lcao_ints.get2b("<ab||ci>").conj())
		t1new += -0.5*einsum('aemn,einm->ai', self.t2, self.lcao_ints.get2b("<ai||jk>").conj())
		t1new /= eia



		# T2 equation

		t2new = self.lcao_ints.get2b("<ab||ij>")

		TMPbe = Fvv
		TMPbe -= 0.5*einsum("bm,me->be",self.t1,Fov)
		Pabij = einsum("aeij,be->abij",self.t2,TMPbe)
		t2new += Pabij - Pabij.transpose(1,0,2,3)

		TMPmj = Foo 
		TMPmj += 0.5 * einsum("ej,me->mj",self.t1,Fov)
		Pabij = einsum("abim,mj->abij",self.t2,TMPmj)
		t2new -= Pabij + Pabij.transpose(0,1,3,2)
	
		t2new += 0.5*einsum('abmn,mnij->abij', tau, Woooo)
		t2new += 0.5*einsum('efij,abef->abij', tau, Wvvvv)


		TMPmbij = einsum("ei,bmej->mbij",self.t1,self.lcao_ints.get2b("<ai||bj>"))
		Pabij = einsum("am,mbij->abij",self.t1,TMPmbij)
		Pabij += einsum("aeim,mbej->abij",self.t2,Wovvo)
		Pabij -= Pabij.transpose(0,1,3,2)
		Pabij -= Pabij.transpose(1,0,2,3)
		t2new += Pabij

		Pabij = einsum("ei,abej->abij",self.t1,self.lcao_ints.get2b("<ab||ci>"))
		t2new += Pabij
		t2new -= Pabij.transpose(0,1,3,2)


		Pabij = -einsum("am,bmij->abij",self.t1,self.lcao_ints.get2b("<ai||jk>"))
		t2new -= Pabij
		t2new += Pabij.transpose(1,0,2,3)
		
		t2new /= eijab

		return t1new, t2new
















