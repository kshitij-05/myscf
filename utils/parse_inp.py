import numpy as np
from pyscf import gto
from pyscf.gto.basis import parse_gaussian
import json

def parse_atoms(atoms):
	geom = atoms.split(";")
	symbols =[]
	for a in geom:
		at = a.split(" ")
		symbols.append(at[0])
	return symbols


class parse_input:
	def __init__(self, inp_file):
		self.inp_name = inp_file
		self.task = "scf"
		self.atoms = ""
		self.interface = "pyscf"
		self.basis = ""
		self.custom_basis = ""
		self.scf = {}
		self.unit= "A"
		self.charge = 0
		self.multiplicity = 1  
		self.scf_convergence = 1e-10
		self.use_abcd = False
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
		if self.custom_basis !="":
			symbols = parse_atoms(self.atoms)
			c_basis = {}
			for s,symbol in enumerate(symbols):
				c_basis[symbol]	= parse_gaussian.load(self.custom_basis,symbol)
			mol.basis = c_basis
			
		else:
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
		self.unit = self.if_exist_return("unit")
		self.x2c = self.if_exist_return("x2c")
		self.interface = self.if_exist_return("interface")

		if self.has_key("basis"):
			self.basis = self.if_exist_return("basis")

		if self.has_key("charge"):
			self.charge = int(self.if_exist_return("charge"))

		self.multiplicity = int(self.if_exist_return("multiplicity"))

		if self.has_key("task"):
			self.task = self.if_exist_return("task")

		if self.has_key("abcd"):
			self.use_abcd = self.if_exist_return("abcd")

		if self.has_key("custom_basis"):
			self.custom_basis = self.if_exist_return("custom_basis")


		if self.x2c:
			if "x2c" in self.scf:
				self.x2c = self.scf["x2c"]
		if "scf_convergence" in content:
			self.scf_convergence = float(content["scf_convergence"])
			













