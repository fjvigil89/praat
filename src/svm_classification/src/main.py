import os, sys
from bin import run_baseline
from bin import run_baseline_multiclass
from bin import run_compute_csvfeatures_opensmile
from src import Crea_list_kfold

sys.path.append('bin')
sys.path.append('../data')
os.chdir('../')

base_train = 'thalento'  # 'Saarbruecken'     ''  # 'AVFAD' #   #
base_test = 'thalento'  # 'thalento'
tipo_signal = 'D10_CANTO'  # 'D8_LECTURA'
tipo_signal_test = 'D10_CANTO'  # 'phrase_both'

# run_compute_csvfeatures_opensmile.main("data/audio/thalento_lectura_10/", "data/features/thalento/")

# Crea_list_kfold.selecciona_conjuntos_Cross_validation('binaria', 'GroupKFold', 5, base_train, tipo_signal)
Crea_list_kfold.selecciona_conjuntos_Cross_validation('binaria', 'StratifiedGroupKFold', 5, base_train, tipo_signal)

# Crea_list_kfold.selecciona_conjuntos_Cross_validation('multiclases2binarias', 'StratifiedGroupKFold', 5, base,
# tipo_signal) Crea_list_kfold.selecciona_conjuntos_Cross_validation('multiclases', 'StratifiedGroupKFold', 5, base,
# tipo_signal)

run_baseline.crea_paquetes('data/lst/' + base_train, 5, tipo_signal, 'nuevo', 'binaria')
# run_baseline.crea_paquetes('data/lst/' + base, 5, tipo_signal, 'nuevo', 'multiclases2binarias')
# run_baseline.crea_paquetes('data/lst/' + base, 5, tipo_signal, 'nuevo', 'multiclases')

run_baseline.main_with_thalento('data/lst/' + base_train, 'data/lst/' + base_test, 5, tipo_signal, tipo_signal_test,
                                'viejo', 'binaria')
# run_baseline.main('data/lst/' + base, 1, tipo_signal, 'viejo', 'binaria')
# run_baseline.main('data/lst/' + base, 5, tipo_signal, 'viejo', 'multiclases2binarias')
# run_baseline_multiclass.main('data/lst/' + base, 5, tipo_signal, True)
