from scf.mean_field import gmf
from pyscf import cc as pycc 
from cc.so_cc import ccsd

class task:
	def __init__(self, inp):
		self.inp =inp

	def eval_task(self):

		if self.inp.task == "scf":
			mf = gmf(self.inp).do_scf()

		if self.inp.task == "ccsd":
			mf = gmf(self.inp).do_scf()
			if self.inp.scf["type"] != "x2c":
				mycc = pycc.CCSD(mf)
				mycc.kernel()

			elif self.inp.scf["type"] == "x2c":
				mycc = ccsd(mf,self.inp)
				mycc.kernel()