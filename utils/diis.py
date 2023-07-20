import numpy as np


class diis:
	def __init__(self,nmats,ndiis=6):
		self.nmats = nmats 
		self.ndiis = ndiis
		self.error_vec = []



