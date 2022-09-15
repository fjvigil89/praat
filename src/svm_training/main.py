import os, sys


from svm_training import run_baseline
from svm_training import run_baseline_multiclass
from svm_training import run_compute_csvfeatures_opensmile
from svm_training import Crea_list_kfold


# sys.path.append('bin')
# sys.path.append('../data')
# os.chdir('../')


base_train = 'thalento'  # 'Saarbruecken'     ''  # 'AVFAD' #   #
base_test = 'thalento'  # 'thalento'
tipo_signal = "D8_LECTURA" #'LECTURA'  # 'D8_LECTURA'
tipo_signal_test = 'LECTURA'  # 'phrase_both'
path_database="data/audio/thalento/"
path_features="data/features/thalento/"
path_list_fold="data/lst/"


# def run_compute():   
#     #No funciona, preguntarle a Flavio 
#     run_compute_csvfeatures_opensmile.main(path_database, path_features)

### Esta función es para crear los kfold en base a un metadata.xlsx de la base de datos ###
def crea_list_kfold():
    Crea_list_kfold.selecciona_conjuntos_Cross_validation('binaria', 'StratifiedGroupKFold', 5, base_train, tipo_signal)

# def main_with_thalento():
#     print("tested")
#     run_baseline.main_with_thalento(path_list_fold + base_train, path_list_fold + base_test, 5, tipo_signal, tipo_signal_test, 'viejo', 'binaria')


# def crea_paquetes():    
#     run_baseline.crea_paquetes(path_list_fold + base_train, 5, tipo_signal, 'nuevo', 'binaria')

def feature_smile():    
    run_baseline.feature_smile(path_list_fold + base_train, 5, tipo_signal, 'nuevo', 'binaria')

def train_svm():    
    run_baseline.svm_binario(path_list_fold + base_train, 5, tipo_signal, 'binaria')

def train_svm_flavio():    
    run_baseline.main(path_list_fold + base_train, 5, tipo_signal, 'binaria')
    
def tiempo_total():
    run_baseline.tiempo_total(path_list_fold + base_train, 5, tipo_signal)
