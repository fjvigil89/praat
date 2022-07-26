import pandas as pd
from pandas import ExcelWriter
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

def crea_metadata_voice():
    root = 'D:\Gabriel\Bases\Voice'
    ID = []; Age = []; Gender = []; Diagnosis = []; VHI = []; RSI = []
    for name in os.listdir(root):
        aux = name.split('-')
        if 'info.txt' in aux:
            with open(root + '/' + name) as file:
                index = 0
                for linea in file:
                    linea = linea.replace('\n', '')
                    if ':' in linea:
                        linea = linea.replace(':', '')
                    if '\t' in linea:
                        temp = linea.split('\t')
                    else:
                        temp = linea.split(' ')
                    if temp[0] == 'ID': ID.append(temp[1]); index = index + 1
                    if temp[0] == 'Age': Age.append(temp[1]); index = index + 1
                    if temp[0] == 'Gender': Gender.append(temp[1]); index = index + 1
                    if temp[0] == 'Diagnosis':
                        if temp[1] == 'healthy':
                            Diagnosis.append('control')
                        else: Diagnosis.append(temp[1])
                        index = index + 1
                    if temp[0] == 'Voice Handicap Index (VHI) Score':
                        VHI.append(temp[1])
                    if temp[0] == 'Reflux Symptom Index (RSI) Score':
                        RSI.append(temp[1])
                    print(linea)
                if index != 4:
                    print('problemas')
                    break

    df = pd.DataFrame({'File ID': ID,
                       'Age': Age,
                       'Sex': Gender,
                       'CMVD-I Dimension 1 (word system)': Diagnosis,
                       'Voice Handicap Index (VHI) Score': VHI,
                       'Reflux Symptom Index (RSI) Score': RSI})
    df = df[['File ID', 'Age', 'Sex', 'CMVD-I Dimension 1 (word system)', 'Voice Handicap Index (VHI) Score', 'Reflux Symptom Index (RSI) Score']]
    writer = ExcelWriter('D:\Gabriel\Trabajo\patologias/baseline\data/lst/VOICE/VOICE_metadata.xlsx')
    df.to_excel(writer, 'VOICE', index=False)
    writer.save()

def load_metadata(name_base, clases = 'binaria'):
    path_metadata = './data/lst/' + name_base + '/' + name_base + '_metadata.xlsx'
    if name_base == 'Saarbruecken':
        df = pd.read_excel(path_metadata, sheet_name='sheet1')
    elif name_base == 'AVFAD':
        df = pd.read_excel(path_metadata, sheet_name=name_base)
    elif name_base == 'VOICED':
        df = pd.read_excel(path_metadata, sheet_name=name_base)
    dict_info_signal = {}
    for ind, row in df.iterrows():
        id_detail = str(row['CMVD-I Dimension 1 (numeric system)'])
        detail = row['CMVD-I Dimension 1 (word system)']
        if id_detail == '0':
            patolo = 'n'
        else:
            patolo = 'p'

        if name_base == 'Saarbruecken':
            dict_info_signal[str(row['File ID'])] = {'spk': str(row['SPEAKER']), 'Path': patolo, 'age': row['AGE'],
                    'gender': row['GENDER'], 'tipo': row['CMVD-I Dimension 1 (word system)'], 'Base': 'Saarbruecken',
                    'ids_pathological': id_detail, 'DETAIL_grupo': row['CMVD-I word class'],
                    'grupo_id': row['CMVD-I numeric class']}
        elif name_base == 'AVFAD' or name_base == 'VOICED':
                dict_info_signal[row['File ID']] = {'spk': row['File ID'], 'Path': patolo, 'age': row['Age'],
                    'gender': row['Sex'], 'tipo': detail, 'Base': name_base, 'ids_pathological': id_detail,
                    'DETAIL_grupo': row['CMVD-I word class'], 'grupo_id': row['CMVD-I numeric class']}
        #elif name_base == 'VOICE':
        #    dict_info_signal[str(row['File ID'])] = {'spk': row['File ID'], 'Path': patolo, 'age': row['Age'],
        #                                    'gender': row['Sex'], 'tipo': row['CMVD-I Dimension 1 (word system)'], 'Base': 'VOICE',
        #                                    'DETAIL_grupo': row['DETAIL_grupo'], 'grupo_id': row['CMVD-I_grupo']}

    return dict_info_signal

def main(name_base, clases = 'binaria'):
    print(pathlib.Path(__file__).parent.absolute())
    dict_info_signal = {}
    if name_base == 'Saarbruecken' or name_base == 'AVFAD' or name_base == 'VOICED':
        dict_info_signal = load_metadata(name_base, clases)
    elif name_base == 'Saarbruecken_AVFAD':
        dict_info_signal = load_metadata('Saarbruecken', clases)
        dict_AVFAD = load_metadata('AVFAD', clases)
        dict_info_signal.update(dict_AVFAD)
    elif name_base == 'Saarbruecken_AVFAD_VOICED':
        dict_info_signal = load_metadata('Saarbruecken', clases)
        dict_AVFAD = load_metadata('AVFAD', clases)
        dict_VOIVED = load_metadata('VOICED', clases)
        dict_info_signal.update(dict_AVFAD)
        dict_info_signal.update(dict_VOIVED)

    return dict_info_signal

if __name__ == '__main__':
    #path_metadata = 'D:\Gabriel\Bases\Saarbruecken\metadata.xls'
    #arregla_base(path_metadata)

    #path_metadata = 'D:\Gabriel\Bases\Saarbruecken\metadata_new.xlsx'
    #main(path_metadata)
    crea_metadata_voice()