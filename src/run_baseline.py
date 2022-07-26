import numpy as np
import os, sys


sys.path.append('bin')
sys.path.append('../data')

from bin import run_baseline
from bin import run_baseline_multiclass

os.chdir('../')

#run_baseline.crea_paquetes('data/lst/Saarbruecken_G2', 5, 'phrase_both', 'nuevo', 'binaria')
#run_baseline.crea_paquetes('data/lst/Saarbruecken_G2', 5, 'phrase_both', 'nuevo', 'multiclases2binarias')
#run_baseline.crea_paquetes('data/lst/Saarbruecken_G2', 5, 'phrase_both', 'nuevo', 'multiclases')


#run_baseline.main('data/lst/Saarbruecken_G2', 5, 'phrase_both', 'viejo', 'binaria')
run_baseline.main('data/lst/Saarbruecken_G2', 5, 'phrase_both', 'viejo', 'multiclases2binarias')
run_baseline_multiclass.main('data/lst/Saarbruecken_G2', 5, 'phrase_both')
