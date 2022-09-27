from ast import And
from asyncio.windows_events import NULL
from functools import total_ordering
from genericpath import isdir, isfile
from multiprocessing.dummy import Array
import sys, json, os, pickle, time
import datetime
# sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.svm import SVC
import csv
import opensmile
import mutagen
from mutagen.wave import WAVE
from svm_training import utils
#from utils import compute_score, zscore
from collections import Counter

# function to convert the information into 
# some readable format
def audio_duration(length):
    hours = length // 3600  # calculate in hours
    length %= 3600
    mins = length // 60  # calculate in minutes
    length %= 60
    seconds = length  # calculate in seconds
    
    if mins < 10:
        mins = '0'+str(mins)
    if seconds < 10:
        seconds = '0'+str(seconds)
    return hours, mins, seconds  # returns the duration


def timeSaarbruecken(list_path, path_metadata, label):     
    df = pd.read_excel (path_metadata , sheet_name= label)
    df = df.assign(Time="0:00:00")   
    df = df.assign(allTime="0:00:00") 
    total_time= datetime.timedelta(hours=00,minutes=00,seconds=00)
    
    timepath = 'data/pathology/' + label
    if not os.path.exists(timepath):
        os.makedirs(timepath)
    listPathology=["Abnormalities of the Vocal Fold", "control", "Dysphonia", "INFLAMMATORY CONDITIONS OF THE LARYNX", "NEUROLOGIC DISORDERS AFFECTING VOICE", "OTHER DISORDERS AFFECTING VOICE", "Recurrent Paralysis", "Spasmodic Dysphonia", "SYSTEMIC CONDITIONS AFFECTING VOICE"]
    timeAbnomalie= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeControl= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeDysphonia= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeInflammatory= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeNeurology= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeOther= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeRecurrent= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeSpasmodic= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeSystemic= datetime.timedelta(hours=00,minutes=00,seconds=00)
    listTime=[timeAbnomalie, timeControl,timeDysphonia, timeInflammatory,timeNeurology , timeOther,timeRecurrent , timeSpasmodic, timeSystemic ]
    i=0   
    for item in df['File ID']:
        print(df['PATHOLOGY'][i])
        p= "PATH" if df['PATHOLOGY'][i]=='p' else "NORM"
        g= "hombres" if df['GENDER'][i]=='m' else "mujeres"
        path= list_path+"/"+p+"/"+g        
        timeperitem= datetime.timedelta(hours=00,minutes=00,seconds=00)
        index = df[df['File ID']==item].index.tolist()
        for r, d, n in os.walk(path):    
            for file in n:                
                if int(file.split("-")[0]) == item:
                    if "-a_h.wav" in file  or "-a_l.wav" in file or "-a_lhl.wav" in file or "-a_n.wav" in file or "-i_h.wav" in file or "-i_l.wav" in file or "-i_lhl.wav" in file or "-i_n.wav" in file or "-phrase.wav" in file or "-u_h.wav" in file or "-u_l.wav" in file or "-u_lhl.wav" in file or "-u_n.wav" in file:
                        path = r + '/' + file                                             
                        audio = WAVE(path)
                        audio_info = audio.info
                        length = int(audio_info.length)
                        hours, mins, seconds = audio_duration(length)
                        time = datetime.timedelta(hours=int(hours),minutes=int(mins),seconds=int(seconds))
                        total_time=total_time+time
                        timeperitem=timeperitem+time
                        j=0
                        for pathology in listPathology:
                            if pathology == df["DETAIL_grupo"][i]:
                                listTime[j]= listTime[j] + time                            
                                break
                            j=j+1
                        #print(file,' Total Duration: {}:{}:{}'.format(hours, mins, seconds))
                                                                           
        
        df['Time'][i]=str(timeperitem)
        print(item, df['Time'][i])       
        i=i+1
    
    df['allTime']=" "
    df["timeAbnomalie"]=" "
    df["timeControl"]=" "
    df["timeDysphonia"]=" "
    df["timeInflammatory"]=" "
    df["timeOther"]=" "
    df["timeSpasmodic"]=" "
    df["timeNeurology"]=" "
    df["timeRecurrent"]=" "
    df["timeSystemic"]=" "
       
    
    df['allTime'][0]=str(total_time)
    df["timeAbnomalie"][0]=str(listTime[0])
    df["timeControl"][0]=str(listTime[1])
    df["timeDysphonia"][0]=str(listTime[2])
    df["timeInflammatory"][0]=str(listTime[3])
    df["timeNeurology"][0]=str(listTime[4])
    df["timeOther"][0]=str(listTime[5])
    df["timeRecurrent"][0]=str(listTime[6])
    df["timeSpasmodic"][0]=str(listTime[7])
    df["timeSystemic"][0]=str(listTime[8])    
   
     
    df.to_excel(str(timepath)+'/'+str(label)+'.xlsx', sheet_name=label, index=False)
    print("tiempo total de las grabaciones de "+ label+":",total_time)   

def featureSaarbruecken(list_path, kfold, audio_type, label):
    clases ="binaria"