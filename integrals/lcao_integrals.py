from integrals.ao_integrals import ao_integrals
from integrals.lcao_direct import general
from pyscf import lib
import numpy as np

#einsum = np.einsum
einsum = lib.einsum


def aotomo(mol,mo_coeffs):
	ao = mol.intor("int2e_spinor")

	temp = einsum("ip,ijkl->pjkl",mo_coeffs[0].conj(),ao)
	temp1 = einsum("jq,pjkl->pqkl",mo_coeffs[1],temp)
	temp = einsum("pqkl,kr->pqrl",temp1,mo_coeffs[2].conj())
	temp1 = einsum("pqrl,ls->pqrs",temp,mo_coeffs[3])

	return temp1


def strip_formula(formula):
	strp_form = formula
	strp_form = strp_form.replace("<","") 
	strp_form = strp_form.replace(">","")
	strp_form = strp_form.replace("|","")
	if len(strp_form)==3:
		strp_form = strp_form.replace("f","")
	return strp_form

class lcao_integrals:
	def __init__(self,mf,inp):
		self.mf = mf
		self.inp = inp
		self.dict = {}
		self.ao_integrals = ao_integrals(inp)
		nelec = self.inp.molecule().nelectron
		self.Ci = mf.mo_coeff[:,:nelec]
		self.Ca = mf.mo_coeff[:,nelec:]

		print(f"nocc = {nelec}")
		print(f"nvir = {self.Ca.shape[1]}")

		if inp.scf["type"] != "x2c":
			self.dtype = np.float64
		else:
			self.dtype = np.complex128

	def is_in_dict(self, formula):
		return (formula in self.dict)

	def which_c(self, sp_state):
		if(sp_state in ["i","j","k","l","m","n"]):
			return self.Ci
		elif(sp_state in ["a","b","c","d","e","f"]):
			return self.Ca


	def get1b(self,formula):
		if not self.is_in_dict(formula):
			str_formula = strip_formula(formula)
			nelec = self.inp.molecule().nelectron
			f = np.diag(self.mf.mo_energy)
			assert(len(str_formula)==2)
			result = None 
			
			if(str_formula[0] in ["i","j","k","l","m","n"]):
				result = f[:nelec,:]	
			elif(str_formula[0] in ["a","b","c","d","e","f"]):
				result = f[nelec:,:]
			elif(str_formula[0] in ["p","q","r","s","t"]):
				result = f[:,:]

			if (str_formula[1] in ["i","j","k","l","m","n"]):
				result = result[:,:nelec]
			elif(str_formula[1] in ["a","b","c","d","e","f"]):
				result = result[:,nelec:]
			elif(str_formula[0] in ["p","q","r","s","t"]):
				result = f[:,:]


			result = result.astype(self.dtype)
			self.dict[formula] = result
			return result

		else:
			return self.dict[formula]

	

	def get2b(self,formula):

		# TODO: check weather formula is antisymmetric or not for now assumed anti symmetric

		if not self.is_in_dict(formula):
			strp_form = strip_formula(formula)
			assert(len(strp_form)==4)
			c0 = self.which_c(strp_form[0])
			c1 = self.which_c(strp_form[1])
			c2 = self.which_c(strp_form[2])
			c3 = self.which_c(strp_form[3])

			# <01||23> = <01|23> - <01|32> = (02|13) - (03|12)

			# (02|13)
			int1 = general(self.inp.molecule(),[c0,c2,c1,c3])
			int1 = np.reshape(int1,(c0.shape[1],c2.shape[1],c1.shape[1],c3.shape[1]))
			
			# (03|12)
			int2 = general(self.inp.molecule(),[c0,c3,c1,c2])
			int2 = np.reshape(int2,(c0.shape[1],c3.shape[1],c1.shape[1],c2.shape[1]))

			#int1 = aotomo(self.inp.molecule(),[c0,c2,c1,c3])
			#int2 = aotomo(self.inp.molecule(),[c0,c3,c1,c2])

			# <01||23> = <01|23> - <01|32>
			result = int1.transpose(0,2,1,3) - int2.transpose(0,2,3,1)
			self.dict[formula] = result
			print(f"added {formula} to registry")
			return result

		else:
			return self.dict[formula]

	def get_denom2(self):
		nelec = self.inp.molecule().nelectron
		mo_e_o = self.mf.mo_energy[:nelec]
		mo_e_v = self.mf.mo_energy[nelec:]
		eia = mo_e_o[:,None] - mo_e_v
		eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
		eia = eia.astype(self.dtype)
		eijab = eijab.astype(self.dtype)
		return eia ,eijab










