import json
import os
import sys

import numpy as np

from svm_training import Cross_validation
# import Cross_validation
#from svm_training import Load_metadata as db
from svm_training import utils, db_avfad, db_voiced, db_thalento, db_saarbruecken, Load_metadata as db
# sys.path.append('bin')
# sys.path.append('../data')


def prepara_listas_Cross_validation(sesion, clases = 'binaria'):
    list_muestras = []
    list_clases = []
    list_grupos = []
    dict_clases = {"NORM": '0', "PATH": '1'}
    for j in sesion:
        list_muestras.append(j)
        list_grupos.append(sesion[j]['spk'])
        if sesion[j]['Path'] == 'n':
            list_clases.append('0')
        elif clases == 'binaria':
            list_clases.append('1')
        else:
            aux = sesion[j]['tipo'].split(';')
            if clases == 'multiclases':
                dict_clases["multi_patology"] = '2'
                for i in aux:
                    info = i.split(':')
                    if len(info) > 1 and info[0] not in dict_clases:
                        dict_clases[info[0].replace(' ','')] = info[1].replace(' ','')
            if len(aux) > 2:
                list_clases.append('2')
            else:
                aux = aux[0].split(':')
                list_clases.append(aux[1].replace(' ',''))
            
    return list_muestras, list_clases, list_grupos, dict_clases


def salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, clases, dict_clases, name_base, grabacion):
    fold_train = {}; fold_test = {}; tipo_clases = {}
    tipo = grabacion.replace('_both','')
    fold_train = {"labels": dict_clases, "meta_data": []}
    fold_test = {"labels": dict_clases, "meta_data": []}

    train = fold_train['meta_data']
    test = fold_test['meta_data']

    index = 0
    for i in muestras_train:
        spk = dict_info_signal[i]['spk']
        label = list_clases_train[index]

        index = index + 1
        #gender = 'hombres'
        #if dict_info_signal[i]['gender'] == 'w':
        #   gender = 'mujeres'
        if name_base == 'AVFAD':
            aa = {'path': 'data/audio/' + name_base + '/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
        else:
            aa = {'path': 'data/audio/' + name_base + '/' + str(i) + '-' + tipo + '.wav', 'label': label,
                  'speaker': spk}
        train.append(aa)
    fold_train['meta_data'] = train

    index = 0
    for i in muestras_test:
        spk = dict_info_signal[i]['spk']
        label = list_clases_test[index]

        index = index + 1
        #gender = 'hombres'
        #if dict_info_signal[i]['gender'] == 'w':
        #    gender = 'mujeres'
        if name_base == 'AVFAD':
            aa = {'path': 'data/audio/' + name_base + '/' + str(i) + tipo + '.wav', 'label': label,
              'speaker': spk}
        else:
            aa = {'path': 'data/audio/' + name_base + '/' + str(i) + '-' + tipo + '.wav', 'label': label,
                  'speaker': spk}
        test.append(aa)
    fold_test['meta_data'] = test

    return fold_train, fold_test

### Esta función es para crear los kfold en base a un metadata.xlsx de la base de datos ###
def selecciona_conjuntos_Cross_validation(clases = 'binaria', metodo = 'GroupKFold', particiones = 5, name_base = 'Saarbruecken', grabacion = 'phrase_both'):
    dict_info_signal = db.main('./lst/' + name_base + '/' + name_base + '_metadata.xlsx', name_base)
    list_muestras, list_clases, list_grupos, dict_clases = prepara_listas_Cross_validation(dict_info_signal, clases);
    #elif clases == 'multiclases2binarias' or clases == 'multiclases':

    if metodo == 'GroupKFold':
        dict_fold = Cross_validation.GroupKFold_G(list_muestras, list_clases, list_grupos, particiones)
    elif metodo == 'StratifiedGroupKFold':
        dict_fold = Cross_validation.StratifiedGroupKFold_G(list_muestras, list_clases, list_grupos, particiones)

    ind = 1; camino = './lst/' + name_base
    for i in dict_fold:
        ind_train = np.array(dict_fold[i]['train'])
        ind_test = np.array(dict_fold[i]['test'])
        muestras_train = np.array(list_muestras)[ind_train]
        list_clases_train = np.array(list_clases)[ind_train]
        muestras_test = np.array(list_muestras)[ind_test]
        list_clases_test = np.array(list_clases)[ind_test]

        if clases == 'multiclases2binarias':
            ind_aux = list_clases_train != '0'; list_clases_train[ind_aux] = 1
            ind_aux = list_clases_test != '0'; list_clases_test[ind_aux] = 1

        [fold_train, fold_test] = salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, clases, dict_clases, name_base, grabacion)
        if not os.path.exists(camino):
            os.mkdir(camino + '/')

        with open(camino + '/' + 'train_' + clases + '_' + grabacion + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
            json.dump(fold_train, file, indent=6)
        file.close()
        with open(camino + '/' + 'test_' + clases + '_' + grabacion + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
            json.dump(fold_test, file, indent=6)
        file.close()
        ind = ind + 1

def kford():    
    #db_avfad.kford()
    db_saarbruecken.kford()

if __name__ == '__main__':

    #selecciona_conjuntos_Cross_validation('binaria', 'GroupKFold', 5, 'Saarbruecken','phrase_both')
    #selecciona_conjuntos_Cross_validation('multiclases2binarias', 'StratifiedGroupKFold', 5, 'Saarbruecken', 'phrase_both')
    selecciona_conjuntos_Cross_validation('multiclases', 'StratifiedGroupKFold', 5, 'Saarbruecken', 'phrase_both')

