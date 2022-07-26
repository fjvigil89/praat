import Load_metadata as db
import numpy as np
import os, sys
import Cross_validation
import json
from bin import run_compute_csvfeatures_opensmile

sys.path.append('bin')
sys.path.append('../data')

def prepara_listas_Cross_validation(sesion, clases = 'binaria', multi_sanos = False):
    list_muestras = []
    list_clases = []
    list_grupos = []
    hombres = 0
    if clases == 'binaria':
        dict_clases = {"NORM": '0', "PATH": '1'}
    elif clases == 'multiclases':
        dict_clases = {}
    for j in sesion:
        if clases == 'binaria':
            list_muestras.append(j)
            list_grupos.append(sesion[j]['spk'])
            if sesion[j]['Path'] == 'n':
                list_clases.append('0')
            else:
                list_clases.append('1')
        elif clases == 'multiclases':
            if sesion[j]['Path'] == 'p':
                list_muestras.append(j)
                list_grupos.append(sesion[j]['spk'])
                list_clases.append(sesion[j]['grupo_id'])
                # aux = sesion[j]['DETAIL_grupo']
                dict_clases[sesion[j]['DETAIL_grupo']] = str(sesion[j]['grupo_id'])
            elif multi_sanos and sesion[j]['Path'] == 'n':
                list_muestras.append(j)
                list_grupos.append(sesion[j]['spk'])
                list_clases.append(0)
                # aux = sesion[j]['DETAIL_grupo']
                dict_clases[sesion[j]['DETAIL_grupo']] = str(sesion[j]['grupo_id'])

            
    return list_muestras, list_clases, list_grupos, dict_clases


def salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, clases, dict_clases, grabacion):
    fold_train = {}; fold_test = {}; tipo_clases = {}
    tipo = grabacion.replace('_both','')
    fold_train = {"labels": dict_clases, "meta_data": []}
    fold_test = {"labels": dict_clases, "meta_data": []}

    train = fold_train['meta_data']
    test = fold_test['meta_data']
    #run_compute_csvfeatures_opensmile.main()
    index = 0
    for i in muestras_train:
        spk = dict_info_signal[i]['spk']
        label = str(list_clases_train[index])

        index = index + 1
        #if dict_info_signal[i]['gender'] == 'm':
        #    gender = 'hombres'
        #else:
        #    gender = 'mujeres'
        #if label == '0':
        #    patolo = 'NORM'
        #else:
        #    patolo = 'PATH'
#
        #    #if dict_info_signal[i]['gender'] == 'w':
        ##   gender = 'mujeres'
        #if name_base == 'AVFAD':
        #    aa = {'path': 'data/audio/' + name_base + '/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
        #elif name_base == 'Saarbruecken':
        #    aa = {'path': 'data/audio/' + name_base + '/' + patolo + '/' + gender + '/' + str(i) + '-' + tipo + '.wav', 'label': label,
        #          'speaker': spk}
        if dict_info_signal[i]['Base'] == 'AVFAD':
            aa = {'path': 'data/audio/' + dict_info_signal[i]['Base'] + '/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
        elif dict_info_signal[i]['Base'] == 'Saarbruecken':
            aa = {'path': 'data/audio/' + dict_info_signal[i]['Base'] + '/' + str(i) + '-' + tipo + '.wav', 'label': label,
                  'speaker': spk}
        elif dict_info_signal[i]['Base'] == 'VOICED':
            aa = {'path': 'data/audio/' + dict_info_signal[i]['Base'] + '/' + str(i) + '.wav',
                  'label': label, 'speaker': spk}
            #out = 'data/features/' + dict_info_signal[i]['Base'] + '/'
            #run_compute_csvfeatures_opensmile.main('data/audio/' + dict_info_signal[i]['Base'] + '/', out)

        train.append(aa)
    fold_train['meta_data'] = train

    index = 0
    for i in muestras_test:
        spk = str(dict_info_signal[i]['spk'])
        label = str(list_clases_test[index])

        index = index + 1
        #if dict_info_signal[i]['gender'] == 'm':
        #    gender = 'hombres'
        #else:
        #    gender = 'mujeres'
        #if label == '0':
        #    patolo = 'NORM'
        #else:
        #    patolo = 'PATH'
        #if name_base == 'AVFAD':
        #    aa = {'path': 'data/audio/' + name_base + '/' + str(i) + tipo + '.wav', 'label': label,
        #      'speaker': spk}
        #elif name_base == 'Saarbruecken':
        #    aa = {'path': 'data/audio/' + name_base + '/' + patolo + '/' + gender + '/' + str(i) + '-' + tipo + '.wav', 'label': label,
        #          'speaker': spk}
        if dict_info_signal[i]['Base'] == 'AVFAD':
            aa = {'path': 'data/audio/' + dict_info_signal[i]['Base'] + '/' + str(i) + tipo + '.wav', 'label': label,
                  'speaker': spk}
        elif dict_info_signal[i]['Base'] == 'Saarbruecken':
            aa = {'path': 'data/audio/' + dict_info_signal[i]['Base'] + '/' + str(i) + '-' + tipo + '.wav', 'label': label,
                  'speaker': spk}
        elif dict_info_signal[i]['Base'] == 'VOICED':
            aa = {'path': 'data/audio/' + dict_info_signal[i]['Base'] + '/' + str(i) + '.wav',
                  'label': label, 'speaker': spk}
        test.append(aa)
    fold_test['meta_data'] = test

    return fold_train, fold_test

def leebase(base, clases = 'binaria'):
    sesion = db.main(base, 'binaria')
    dictaux = {}
    hombres_sanos = 0; mujeres_sanos = 0; hombres_p = 0; mujeres_p = 0
    for j in sesion:
        dictaux[sesion[j]['spk']] = sesion[j]
    for j in dictaux:
        if dictaux[j]['gender'] == 'M':
            hombres_sanos = hombres_sanos + 1
        elif dictaux[j]['gender'] == 'F':
            mujeres_sanos = mujeres_sanos + 1
    print('hombres sanos = ' + str(hombres_sanos))
    print('mujeres sanos = ' + str(mujeres_sanos))
    print('TOTAL = ' + str(len(dictaux)))




def selecciona_conjuntos_Cross_validation(clases = 'binaria', metodo = 'GroupKFold', particiones = 5, name_base = 'Saarbruecken', grabacion = 'phrase_both', multi_sanos = True):
    dict_info_signal = db.main(name_base, clases)
    list_muestras, list_clases, list_grupos, dict_clases = prepara_listas_Cross_validation(dict_info_signal, clases, multi_sanos);
    #elif clases == 'multiclases2binarias' or clases == 'multiclases':

    if metodo == 'GroupKFold':
        dict_fold = Cross_validation.GroupKFold_G(list_muestras, list_clases, list_grupos, particiones)
    elif metodo == 'StratifiedGroupKFold':
        dict_fold = Cross_validation.StratifiedGroupKFold_G(list_muestras, list_clases, list_grupos, particiones)

    ind = 1; camino = './data/lst/' + name_base
    for i in dict_fold:
        ind_train = np.array(dict_fold[i]['train'])
        ind_test = np.array(dict_fold[i]['test'])
        muestras_train = np.array(list_muestras)[ind_train]
        list_clases_train = np.array(list_clases)[ind_train]
        muestras_test = np.array(list_muestras)[ind_test]
        list_clases_test = np.array(list_clases)[ind_test]


        [fold_train, fold_test] = salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, clases, dict_clases, grabacion)
        if not os.path.exists(camino):
            os.mkdir(camino + '/')

        with open(camino + '/' + 'train_' + clases + '_' + grabacion + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
            json.dump(fold_train, file, indent=6)
        file.close()
        with open(camino + '/' + 'test_' + clases + '_' + grabacion + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
            json.dump(fold_test, file, indent=6)
        file.close()
        ind = ind + 1


if __name__ == '__main__':

    #selecciona_conjuntos_Cross_validation('binaria', 'GroupKFold', 5, 'Saarbruecken','phrase_both')
    #selecciona_conjuntos_Cross_validation('multiclases2binarias', 'StratifiedGroupKFold', 5, 'Saarbruecken', 'phrase_both')
    selecciona_conjuntos_Cross_validation('multiclases', 'StratifiedGroupKFold', 5, 'Saarbruecken', 'phrase_both')

