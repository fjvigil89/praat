import os, sys


from svm_training import run_baseline
from svm_training import run_baseline_multiclass
from svm_training import run_compute_csvfeatures_opensmile
from svm_training import Crea_list_kfold


# sys.path.append('bin')
# sys.path.append('../data')
# os.chdir('../')


# base_train = 'thalento'  # 'Saarbruecken'     ''  # 'AVFAD' #   #
# base_test = 'thalento'  # 'thalento'
# tipo_signal = "D8_LECTURA" #'LECTURA'  # 'D8_LECTURA'
# tipo_signal_test = 'LECTURA'  # 'phrase_both'
# path_database="data/audio/thalento"
# path_features="data/features/thalento"
# path_list_fold="data/lst/"
# path_metadata="data/pathology/thalento_metadata.xlsx"

# base_train = "VOICED"   #'thalento'  # 'Saarbruecken'     ''  # 'AVFAD' #   #
# base_test = 'VOICED'  # 'thalento'
# tipo_signal = "phrase_both" #"D8_LECTURA" #'LECTURA'  # 'D8_LECTURA'
# tipo_signal_test = "phrase_both" #'LECTURA'  # 'phrase_both'
# path_database="data/audio/VOICED" #thalento/
# path_features="data/features/VOICED"  #thalento/
# path_list_fold="data/lst/"
# path_metadata="data/pathology/VOICED_metadata.xlsx"

####### No esta funcionando las listas para sacar los parametros
# base_train = "AVFAD"   #'thalento'  # 'Saarbruecken'     ''  # 'AVFAD' #   #
# base_test = 'AVFAD'  # 'thalento'
# tipo_signal = "phrase_both" #"D8_LECTURA" #'LECTURA'  # 'D8_LECTURA'
# tipo_signal_test = "phrase_both" #'LECTURA'  # 'phrase_both'
# path_database="data/audio/AVFAD" #thalento/
# path_features="data/features/AVFAD"  #thalento/
# path_list_fold="data/lst/"
# path_metadata="data/pathology/AVFAD_metadata.xlsx"

### listo
base_train = "Saarbruecken" 
base_test = 'Saarbruecken'  
tipo_signal = "phrase_both" 
tipo_signal_test = "phrase_both"
path_database="data/audio/Saarbruecken"
path_features="data/features/Saarbruecken"
path_list_fold="data/lst/"
path_metadata="data/pathology/Saarbruecken_metadata.xlsx"



# def run_compute():   
#     #No funciona, preguntarle a Flavio 
#     run_compute_csvfeatures_opensmile.main(path_database, path_features)

### Esta función es para crear los kfold en base a un metadata.xlsx de la base de datos ###
def crea_list_kfold():
    Crea_list_kfold.kford()
    #Crea_list_kfold.selecciona_conjuntos_Cross_validation('multiclases', 'StratifiedGroupKFold', 5, base_train, tipo_signal)
    #Crea_list_kfold.selecciona_conjuntos_Cross_validation('binaria', 'GroupKFold', 5, 'Saarbruecken','phrase_both')
    #Crea_list_kfold.selecciona_conjuntos_Cross_validation('multiclases2binarias', 'StratifiedGroupKFold', 5, 'Saarbruecken', 'phrase_both')
    #Crea_list_kfold.selecciona_conjuntos_Cross_validation('multiclases', 'StratifiedGroupKFold', 5, 'Saarbruecken', 'phrase_both')

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


def tiempo_total_pathology():
    run_baseline.tiempo_total_pathology(path_database, path_metadata)

def tiempo_total_audio():
    run_baseline.tiempo_total_audio(path_database)

def clustering():
    run_baseline.clustering(path_list_fold + base_train, 5, tipo_signal, 'binaria')