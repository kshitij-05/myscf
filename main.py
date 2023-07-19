import sys
from utils.parse_inp import parse_input 
from task_manager import task


input_filename = sys.argv[1]

def rmf(input_filename):
	inp = parse_input(input_filename)
	inp.parse()
	tx = task(inp)
	tx.eval_task()

rmf(input_filename)


