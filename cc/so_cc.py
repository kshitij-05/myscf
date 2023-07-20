from pyscf import lib
import numpy as np
from integrals.lcao_integrals import lcao_integrals
from cc import intermidiates as imd

#einsum = np.einsum
einsum = lib.einsum

def make_tau(t2, t1a, t1b, fac=1, out=None):
	t1t1 = einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
	t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
	tau1 = t1t1 - t1t1.transpose(0,1,3,2)
	tau1 += t2
	return tau1

class eris_:
	def __init__(self, mf, lcao_ints):
		self.fock = np.diag(mf.mo_energy)
		self.mo_energy = mf.mo_energy
		self.ovvv = lcao_ints.get2b("<ia||bc>")
		self.oovv = lcao_ints.get2b("<ij||ab>")
		self.ooov = lcao_ints.get2b("<ij||kc>")
		self.oooo = lcao_ints.get2b("<ij||kl>")
		self.vvvv = lcao_ints.get2b("<ab||cd>")
		self.ovov = lcao_ints.get2b("<ia||jb>")


class ccsd:
	def __init__(self,mf,inp):
		self.ref = inp.scf["type"]
		self.lcao_ints = lcao_integrals(mf, inp)
		self.mf = mf
		self.use_abcd = inp.use_abcd
		self.conv = 1e-9
	
	def kernel(self):

		print("\n Beginning CCSD \n")
		eris = eris_(self.mf, self.lcao_ints)
		t1 = np.zeros_like(self.lcao_ints.get1b("<i|f|a>"))
		t2 = np.zeros_like(self.lcao_ints.get2b("<ij||ab>"))

		error_max_size = 8
		diis_check = 1
		t1_old = t1.copy()
		t2_old = t2.copy()
		T1Set = [t1.copy()]
		T2Set = [t2.copy()]
		errors = []


		OLDCC  = 0
		for icc in range(80):
			t1,t2 = self.update_amps(t1,t2,eris,False)
			ECCSD = self.energy(t1,t2)
			DECC = abs(ECCSD - OLDCC)
			if DECC < self.conv:
				print(f"CCSD is converged, {ECCSD}")
				print("TOTAL ITERATIONS: ", icc)
				break

			print("E corr: {0:.12f}".format(ECCSD), "a.u.", '\t', "DeltaE: {0:.12f}".format(DECC))
			# Appending DIIS vectors to T1 and T2 set
			T1Set.append(t1.copy())
			T2Set.append(t2.copy())
			# calculating error vectors
			error_t1 = (T1Set[-1] - t1_old).ravel()
			error_t2 = (T2Set[-1] - t2_old).ravel()
			errors.append(np.concatenate((error_t1, error_t2)))
			t1_old = t1.copy()
			t2_old = t2.copy()
			if icc >= diis_check:
				# size limit of DIIS vector
				if (len(T1Set) > error_max_size + 1):
					del T1Set[0]
					del T2Set[0]
					del errors[0]
				error_size = len(T1Set) - 1
				# create error matrix B_mat
				B_mat = np.ones((error_size + 1, error_size + 1)) * -1
				B_mat[-1, -1] = 0
				for a1, b1 in enumerate(errors):
					B_mat[a1, a1] = np.dot(b1.real, b1.real)
					for a2, b2 in enumerate(errors):
						if a1 >= a2: continue
						B_mat[a1, a2] = np.dot(b1.real, b2.real)
						B_mat[a2, a1] = B_mat[a1, a2]
				B_mat[:-1, :-1] /= np.abs(B_mat[:-1, :-1]).max()
				# create zero vector
				zero_vector = np.zeros(error_size + 1)
				zero_vector[-1] = -1
				# getting coefficients
				coeff = np.linalg.solve(B_mat, zero_vector)
				# getting extrapolated amplitudes
				t1 = np.zeros_like(t1_old)
				t2 = np.zeros_like(t2_old)
				for i in range(error_size):
					t1 += coeff[i] * T1Set[i + 1]
					t2 += coeff[i] * T2Set[i + 1]
				# Save extrapolated amplitudes to t_old amplitudes
				t1_old = t1.copy()
				t2_old = t2.copy()
				OLDCC = ECCSD

	def update_amps(self, t1, t2, eris, update_t1 = True):


		fov = self.lcao_ints.get1b("<i|f|a>")
		foo = self.lcao_ints.get1b("<i|f|i>")
		fvv = self.lcao_ints.get1b("<a|f|a>")
		mo_e_o = np.diag(foo)
		mo_e_v = np.diag(fvv)

		tau = make_tau(t2, t1, t1)
		tau_tilde = make_tau(t2, t1, t1,fac=0.5)

		# update Fae
		Fvv = fvv - 0.5*einsum('me,ma->ae',fov, t1)
		Fvv += einsum('mf,amef->ae', t1, self.lcao_ints.get2b("<ia||bc>").transpose(1,0,3,2))
		Fvv -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, self.lcao_ints.get2b("<ij||ab>"))


		# update Fmi
		Foo = foo + 0.5*einsum('me,ie->mi',fov, t1)
		Foo += einsum('ne,mnie->mi', t1, self.lcao_ints.get2b("<ij||ka>"))
		Foo += 0.5*einsum('inef,mnef->mi', tau_tilde, self.lcao_ints.get2b("<ij||ab>"))


		# update Fme
		Fov = fov + einsum('nf,mnef->me', t1, self.lcao_ints.get2b("<ij||ab>"))


		# update Wmnij
		tmp = einsum('je,mnie->mnij', t1, self.lcao_ints.get2b("<ij||ka>"))
		Woooo = self.lcao_ints.get2b("<ij||kl>") + tmp - tmp.transpose(0,1,3,2)
		Woooo += 0.25*einsum('ijef,mnef->mnij', tau, self.lcao_ints.get2b("<ij||ab>"))

		# update Wmbej
		Wovvo  = einsum('jf,mbef->mbej', t1, self.lcao_ints.get2b("<ia||bc>"))
		Wovvo += einsum('nb,mnej->mbej', t1, self.lcao_ints.get2b("<ij||ka>").transpose(0,1,3,2))
		Wovvo -= 0.5*einsum('jnfb,mnef->mbej', t2, self.lcao_ints.get2b("<ij||ab>"))
		Wovvo -= einsum('jf,nb,mnef->mbej', t1, t1, self.lcao_ints.get2b("<ij||ab>"))
		Wovvo -= self.lcao_ints.get2b("<ia||jb>").transpose(0,1,3,2)


		# Move energy terms to the other side
		Fvv -= self.lcao_ints.get1b("<a|f|a>")
		Foo -= self.lcao_ints.get1b("<i|f|i>")

		# T1 equation
		t1new  =  einsum('ie,ae->ia', t1, Fvv)
		t1new += -einsum('ma,mi->ia', t1, Foo)
		t1new +=  einsum('imae,me->ia', t2, Fov)
		t1new += -einsum('nf,naif->ia', t1, self.lcao_ints.get2b("<ia||jb>"))
		t1new += -0.5*einsum('imef,maef->ia', t2, self.lcao_ints.get2b("<ia||bc>"))
		t1new += -0.5*einsum('mnae,mnie->ia', t2, self.lcao_ints.get2b("<ij||ka>"))
		t1new += fov.conj()

		# T2 equation
		Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
		tmp = einsum('ijae,be->ijab', t2, Ftmp)
		t2new = tmp - tmp.transpose(0,1,3,2)
		Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
		tmp = einsum('imab,mj->ijab', t2, Ftmp)
		t2new -= tmp - tmp.transpose(1,0,2,3)
		t2new += self.lcao_ints.get2b("<ij||ab>").conj()
		t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
		
		tmp = einsum('imae,mbej->ijab', t2, Wovvo)
		tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, self.lcao_ints.get2b("<ia||jb>"))
		tmp = tmp - tmp.transpose(1,0,2,3)
		tmp = tmp - tmp.transpose(0,1,3,2)
		t2new += tmp
		tmp = einsum('ie,jeba->ijab', t1, self.lcao_ints.get2b("<ia||bc>").conj())
		t2new += (tmp - tmp.transpose(1,0,2,3))
		tmp = einsum('ma,ijmb->ijab', t1, self.lcao_ints.get2b("<ij||ka>").conj())
		t2new -= (tmp - tmp.transpose(0,1,3,2))



		# PPL term
		if self.use_abcd:
			
			# update Wabcd
			tmp = einsum('mb,mafe->bafe', t1, self.lcao_ints.get2b("<ia||bc>"))
			Wvvvv = self.lcao_ints.get2b("<ab||cd>") - tmp + tmp.transpose(1,0,2,3)
			Wvvvv += einsum('mnab,mnef->abef', tau, 0.25*self.lcao_ints.get2b("<ij||ab>"))
			t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)




		eia = mo_e_o[:,None] - mo_e_v
		eijab = lib.direct_sum('ia,jb->ijab', eia, eia)


		return t1new/eia, t2new/eijab

	def energy(self, t1, t2):
		fov = self.lcao_ints.get1b("<i|f|a>")
		e = einsum('ia,ia', fov, t1)
		e += 0.25*np.einsum('ijab,ijab', t2, self.lcao_ints.get2b("<ij||ab>"))
		e += 0.5 *np.einsum('ia,jb,ijab', t1, t1, self.lcao_ints.get2b("<ij||ab>"))
		if abs(e.imag) > 1e-5:
			print('Non-zero imaginary part found in GCCSD energy %s', e)
		return e.real




