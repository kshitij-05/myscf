import numpy as np
from pyscf import gto
import json

class parse_input:
	def __init__(self, inp_file):
		self.inp_name = inp_file
		self.task = "scf"
		self.atoms = ""
		self.interface = "pyscf"
		self.basis = ""
		self.scf = {}
		self.unit= "A"
		self.charge = 0
		self.multiplicity = 1  
		self.scf_convergence = 1e-10
		self.content = None

	def if_exist_return(self, key):
		if key in  self.content:
			return self.content[key]

	def has_key(self, key):
		if key in self.content:
			return True
		else:
			return False

	def molecule(self):
		mol = gto.M()
		mol.atom = self.atoms
		mol.basis = self.basis
		mol.charge = self.charge
		mol.spin = self.multiplicity -1 #spin in 2s not 2s+1
		if "verbose" in self.content:
			mol.verbose = int(self.content["verbose"])
		else:
			mol.verbose=4
		mol.build()
		return mol

	def parse(self):
		with open(self.inp_name,'r') as f:
			content = json.loads(f.read(), strict=False)
		self.content = content
		self.atoms = content["atoms"]
		self.scf = content["scf"]
		self.basis = content["basis"]
		self.unit = self.if_exist_return("unit")
		self.x2c = self.if_exist_return("x2c")
		self.interface = self.if_exist_return("interface")

		if self.has_key("charge"):
			self.charge = int(self.if_exist_return("charge"))

		self.multiplicity = int(self.if_exist_return("multiplicity"))

		if self.has_key("task"):
			self.task = self.if_exist_return("task")

		if self.x2c:
			if "x2c" in self.scf:
				self.x2c = self.scf["x2c"]
		if "scf_convergence" in content:
			self.scf_convergence = float(content["scf_convergence"])
			













