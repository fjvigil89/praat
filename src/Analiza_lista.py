import matplotlib
import pandas as pd

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from itertools import chain
from collections import defaultdict
from collections import Counter
import Load_metadata as db

from math import radians
from matplotlib.table import Table

camino = 'D:\Gabriel\Bases\Saarbruecken\Saarbruecken_list_NEW\\'

def carga_Info_Saarbruecken():
    f = open(camino + 'info_Key_Saarbruecken.json')
    info_locutores = json.load(f)
    f.close()
    f = open(camino + 'info_sesiones_spk_Saarbruecken.json')
    info_sesiones = json.load(f)
    f.close()
    return [info_locutores, info_sesiones]

def main(metodo, list_path, kfold, audio_type):
    if audio_type == 'phrase_both':
        info_lista_binaryClass(metodo, list_path, kfold, audio_type)
    elif audio_type == 'phrase_4clase':
        info_lista_4Class(metodo, list_path, kfold, audio_type)



def info_lista_binaryClass(metodo, name_base, kfold, audio_type):
    #[info_locutores, info_sesiones] = carga_Info_Saarbruecken()
    dict_info_signal = db.main('../data/lst/' + name_base + '/' + name_base + '_metadata.xlsx')
    label = os.path.basename(list_path)
    # base = 'D:\Gabriel\Bases\Saarbruecken\Saarbruecken_list\\'
    base = camino + metodo + '\\'
    spk_fold_info = {}
    fig1 = plt.figure(metodo, figsize=(10, 10))
    column_labels = ["NORM", "PATH", "Total"]
    row_Labels = ["Female ", "Male", "signals"]
    tipo = ['train', 'test']
    index_grafica = 1
    spk = {}
    spk_total = {}
    spk_fold_W = []
    for sesion in tipo:
        spk_total.clear()
        spk_fold_W.clear()
        if sesion == 'test':
            index_grafica = 11
        for k in range(0, kfold):
            files = []
            labels = []
            genero = []
            spk_W = 0
            trainlist = base + '/' + sesion + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
            trainlist = camino + carpeta + '/' + 'train_' + clases + '_' + grabacion + '_meta_data_fold' + str(ind) + '.json'
            with open(trainlist, 'r') as f:
                data = json.load(f)
                for item in data['meta_data']:
                    datos = item['path'].split('/')
                    if item['label'] == 'NORM':
                        labels.append(0)
                    else:
                        labels.append(1)
                    if datos[4] == 'mujeres':
                        genero.append(0)
                    else:
                        genero.append(1)
                    aux = datos[5].split('-')
                    aux = aux[0]
                    if info_sesiones[aux] in spk:
                        aa = spk[info_sesiones[aux]]
                        aa.append(aux)
                        spk[info_sesiones[aux]] = aa

                    else:
                        spk[info_sesiones[aux]] = [aux]
                    aa = info_locutores[info_sesiones[aux]]
                    if aa['gender'] == 'W':
                        spk_W = spk_W + 1

                    if info_sesiones[aux] in spk_total:
                        aa = spk_total[info_sesiones[aux]]
                        aa.append(aux)
                        spk_total[info_sesiones[aux]] = aa
                    else:
                        spk_total[info_sesiones[aux]] = [aux]
            f.close()

            labels = np.array(labels)
            genero = np.array(genero)
            i_labels_n = np.where(labels == 0);
            i_labels_p = np.where(labels == 1)
            aux = genero[i_labels_n];
            temp = genero[i_labels_p]
            counter = Counter(aux);
            counter2 = Counter(temp)
            w_n = counter[0];
            w_p = counter2[0]
            m_n = counter[1];
            m_p = counter2[1]
            data = [[w_n, w_p, w_n + w_p], [m_n, m_p, m_n + m_p], [w_n + m_n, w_p + m_p, w_n + m_n + w_p + m_p]]
            spk_fold_info['fold_' + str(k + 1)] = len(spk)
            spk_fold_W.append(spk_W)
            df = pd.DataFrame(data, columns=column_labels)
            ax = fig1.add_subplot(4, 5, index_grafica)
            ax.axis('tight')
            ax.axis('off')
            ax.set_title('train_fold_' + str(k + 1))
            spk.clear()
            if index_grafica == 1 or index_grafica == 11:
                tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=row_Labels, loc='center')
            else:
                tabla = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
            index_grafica = index_grafica + 1

        ax = fig1.add_subplot(4, 5, index_grafica + 1)
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Distribución locutores')
        spk_fold_W = np.array(spk_fold_W)
        spk_fold = np.array(list(spk_fold_info.values()));
        spk_fold_M = np.array([])
        spk_fold_M = spk_fold - spk_fold_W
        spk_fold_M = list(spk_fold_M);
        spk_fold_W = list(spk_fold_W);
        spk_fold = list(spk_fold)
        data_spk = [spk_fold_W, spk_fold_M, spk_fold]
        df_spk = pd.DataFrame(data_spk, columns=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'])
        tabla_spk = ax.table(cellText=df_spk.values, colLabels=df_spk.columns, rowLabels=['Female', 'Male', 'speakers'],
                             loc='center')
        tabla_spk.scale(2, 1.5)
        tabla.set_fontsize(28)

        ax = fig1.add_subplot(4, 5, index_grafica + 3)
        ax.pie(spk_fold_info.values(), labels=spk_fold_info.keys(), autopct="%0.1f %%")
        texto = ''
        for i in spk_fold_info:
            texto = texto + i + ' = ' + str(spk_fold_info[i])
        texto = texto + '_#spk_' + str(len(spk))
        ax.set_title('speaker = ' + str(sum(spk_fold_info.values())))

    plt.show()

def info_lista_4Class(metodo, name_base, kfold, audio_type):
    #[info_locutores, info_sesiones] = carga_Info_Saarbruecken()
    dict_info_signal = db.main('../data/lst/' + name_base + '/' + name_base + '_metadata.xlsx')
    label = os.path.basename(name_base)
    # base = 'D:\Gabriel\Bases\Saarbruecken\Saarbruecken_list\\'
    base = '../data/lst/' + name_base + '/'
    spk_fold_info = {}
    fig1 = plt.figure(metodo, figsize=(10, 10))
    column_labels = ["NORM", "PATH", "Total"]
    tipo = ['train', 'test']
    index_grafica = 1
    spk = {}
    spk_total = {}
    spk_fold_W = []
    for sesion in tipo:
        spk_total.clear()
        spk_fold_W.clear()
        if sesion == 'test':
            index_grafica = 11
        for k in range(0, kfold):
            files = []
            labels = []
            genero = []
            spk_W = 0
            trainlist = base + sesion + '_' + metodo + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
            with open(trainlist, 'r') as f:
                data = json.load(f)
                for item in data['meta_data']:
                    datos = item['path'].split('/')
                    if item['label'] == '0':
                        labels.append(0)
                    else:
                        labels.append(1)
                    aux = datos[5].split('-')
                    aux = aux[0]
                    if dict_info_signal['columnas4'][aux] == 'w':
                        genero.append(0)
                    else:
                        genero.append(1)
                    if dict_info_signal[aux] in spk:
                        aa = spk[dict_info_signal[aux]]
                        aa.append(aux)
                        spk[dict_info_signal[aux]] = aa

                    else:
                        spk[dict_info_signal[aux]] = [aux]
                    aa = info_locutores[info_sesiones[aux]]
                    if aa['gender'] == 'W':
                        spk_W = spk_W + 1

                    if info_sesiones[aux] in spk_total:
                        aa = spk_total[info_sesiones[aux]]
                        aa.append(aux)
                        spk_total[info_sesiones[aux]] = aa
                    else:
                        spk_total[info_sesiones[aux]] = [aux]
            f.close()

            labels = np.array(labels)
            genero = np.array(genero)
            i_labels_n = np.where(labels == 0);
            i_labels_p = np.where(labels == 1)
            aux = genero[i_labels_n];
            temp = genero[i_labels_p]
            counter = Counter(aux);
            counter2 = Counter(temp)
            w_n = counter[0];
            w_p = counter2[0]
            m_n = counter[1];
            m_p = counter2[1]
            data = [[w_n, w_p, w_n + w_p], [m_n, m_p, m_n + m_p], [w_n + m_n, w_p + m_p, w_n + m_n + w_p + m_p]]
            spk_fold_info['fold_' + str(k + 1)] = len(spk)
            spk_fold_W.append(spk_W)
            df = pd.DataFrame(data, columns=column_labels)
            ax = fig1.add_subplot(4, 5, index_grafica)
            ax.axis('tight')
            ax.axis('off')
            ax.set_title('train_fold_' + str(k + 1))
            spk.clear()
            if index_grafica == 1 or index_grafica == 11:
                tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=row_Labels, loc='center')
            else:
                tabla = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
            index_grafica = index_grafica + 1

        ax = fig1.add_subplot(4, 5, index_grafica + 1)
        ax.axis('tight')
        ax.axis('off')
        ax.set_title('Distribución locutores')
        spk_fold_W = np.array(spk_fold_W)
        spk_fold = np.array(list(spk_fold_info.values()));
        spk_fold_M = np.array([])
        spk_fold_M = spk_fold - spk_fold_W
        spk_fold_M = list(spk_fold_M);
        spk_fold_W = list(spk_fold_W);
        spk_fold = list(spk_fold)
        data_spk = [spk_fold_W, spk_fold_M, spk_fold]
        df_spk = pd.DataFrame(data_spk, columns=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'])
        tabla_spk = ax.table(cellText=df_spk.values, colLabels=df_spk.columns, rowLabels=['Female', 'Male', 'speakers'],
                             loc='center')
        tabla_spk.scale(2, 1.5)
        tabla.set_fontsize(28)

        ax = fig1.add_subplot(4, 5, index_grafica + 3)
        ax.pie(spk_fold_info.values(), labels=spk_fold_info.keys(), autopct="%0.1f %%")
        texto = ''
        for i in spk_fold_info:
            texto = texto + i + ' = ' + str(spk_fold_info[i])
        texto = texto + '_#spk_' + str(len(spk))
        ax.set_title('speaker = ' + str(sum(spk_fold_info.values())))

    plt.show()

if __name__ == '__main__':
    main('GroupKFold4clases', 'data/lst/Saarbruecken', 5, 'phrase_both')
