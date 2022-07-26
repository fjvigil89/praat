import os, sys
import Crea_list_kfold
from bin import run_baseline
from bin import run_baseline_multiclass
from bin import run_compute_csvfeatures_opensmile

sys.path.append('bin')
sys.path.append('../data')
os.chdir('../')

#'AVFAD' #  Saarbruecken # VOICED # Saarbruecken_AVFAD # Saarbruecken_AVFAD_VOICED
base = 'VOICED'
tipo_signal = 'phrase_both'

#Crea_list_kfold.leebase(base, 'binaria')

#Crea_list_kfold.selecciona_conjuntos_Cross_validation('binaria', 'GroupKFold', 5, base, tipo_signal)
Crea_list_kfold.selecciona_conjuntos_Cross_validation('multiclases', 'StratifiedGroupKFold', 5, base, tipo_signal)

run_baseline.crea_paquetes(base, 5, tipo_signal, 'nuevo', 'multiclases')

#run_baseline.main(base, 5, tipo_signal, 'nuevo', 'binaria')
run_baseline_multiclass.main(base, 5, tipo_signal, True)

