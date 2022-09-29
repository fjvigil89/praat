import pandas as pd
from xlrd import open_workbook
import os
import math
import numpy as np
import pathlib


def carga_perdidos():
    file = open("D:\Gabriel\Bases\Saarbruecken\Saarbruecken_list_NEW\speaker_perdidos.txt")
    sesion = []
    for i in file:
        aux = i.split('=')
        if len(aux) > 1:
            aux = aux[2].split('\n')
            aux = aux[0].split(' ')
            sesion.append(aux[1])
    return sesion


def elimina_perdidos(path_metadata):
    sesion = carga_perdidos()
    df = pd.read_excel(path_metadata, sheet_name='Hoja1')
    index = []
    for ind, row in df.iterrows():
        if ind == 0 or row[0] not in sesion:
            index.append(True)
        else:
            index.append(False)
    new_df = df[index]
    return new_df


def carga_patologias_dict():
    path_metadata = 'D:\Gabriel\Bases\Saarbruecken\Patologias_saarbrueken_AVFAD_V2.xlsx'
    df = pd.read_excel(path_metadata, sheet_name='Sheet1')
    dict_patologias = {}
    for ind, row in df.iterrows():
        if ind != 0 and str(row[0]) != 'nan':
            dict_patologias[row[0]] = [row [1], row[3], row[4]]
    return dict_patologias


def arregla_base(path_metadata):
    new_df = elimina_perdidos(path_metadata)
    dict_patologias = carga_patologias_dict()
    for ind, row in new_df.iterrows():
        if row[7] != 'DETAIL':
            aux = row[7].split(';')
            data = ''
            SVD = 0; AVFAD = 0;
            for i in aux:
                if i[0] == ' ':
                    i = i[1 : ]
                if i == 'control':
                    row[6] = 'SVD'
                else:
                    if str(row[6]) == 'nan':
                        row[6] = ''
                    data = data + dict_patologias[i][0] + ': ' + str(dict_patologias[i][1]) + '; '
                    if SVD == 0 and dict_patologias[i][2] == 'SVD':
                        row[6] = row[6] + dict_patologias[i][2]; SVD = SVD + 1
                    elif AVFAD == 0 and dict_patologias[i][2] == 'AVFAD':
                        row[6] = row[6] + dict_patologias[i][2]; AVFAD = AVFAD + 1
                    elif dict_patologias[i][2] == 'SVD-AVFAD':
                        row[6] = 'SVD-AVFAD'
            if len(data) > 1:
                row[7] = data

    new_df.to_excel('D:\Gabriel\Bases\Saarbruecken\metadata_new_new.xlsx', sheet_name='sheet1', index=False)


def main(path_metadata, label="Sheet1"):    
    #esto solo funciona para THALENTO y VOICED
    print(pathlib.Path(__file__).parent.absolute())
    df = pd.read_excel(path_metadata, sheet_name=label)
    dict_info_signal = {}
    for ind, row in df.iterrows():
        dict_info_signal[row[0]] = {'spk': row[3], 'Path': row [1], 'age': row[5], 'gender': row[4], 'tipo': row[7]}

    return dict_info_signal



if __name__ == '__main__':
    #path_metadata = 'D:\Gabriel\Bases\Saarbruecken\metadata.xls'
    #arregla_base(path_metadata)

    path_metadata = 'D:\Gabriel\Bases\Saarbruecken\metadata_new.xlsx'
    main(path_metadata)