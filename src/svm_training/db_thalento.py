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


def timeTHALENTO(list_path, path_metadata, label):     
    df = pd.read_excel (path_metadata , sheet_name= label)
    df = df.assign(Time="0:00:00")   
    df = df.assign(allTime="0:00:00") 
    total_time= datetime.timedelta(hours=00,minutes=00,seconds=00)
    
    timepath = 'data/pathology/' + label
    if not os.path.exists(timepath):
        os.makedirs(timepath)
    listPathology=["Abnormalities of the Vocal Fold", "Control", "Disfonía", "INFLAMMATORY CONDITIONS OF THE LARYNX", "OTHER DISORDERS AFFECTING VOICE","Recurrent Paralysis"]
    timeAbnomalie= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeControl= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeDysphonia= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeInflammatory= datetime.timedelta(hours=00,minutes=00,seconds=00)    
    timeOther= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeRecurrente= datetime.timedelta(hours=00,minutes=00,seconds=00)
    listTime=[timeAbnomalie, timeControl,timeDysphonia, timeInflammatory, timeOther, timeRecurrente ]
    i=0   
    for item in df['File ID']:    
        path= list_path+"/"+item
        timeperitem= datetime.timedelta(hours=00,minutes=00,seconds=00)                    
        for r, d, n in os.walk(path):    
            for file in n:
                 if '.wav' in file:
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
                        if pathology == df["Group"][i]:
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
    df["timeRecurrent"]=" "
    
    df['allTime'][0]=str(total_time)
    df["timeAbnomalie"][0]=str(listTime[0])
    df["timeControl"][0]=str(listTime[1])
    df["timeDysphonia"][0]=str(listTime[2])
    df["timeInflammatory"][0]=str(listTime[3]) 
    df["timeOther"][0]=str(listTime[4])   
    df["timeRecurrent"][0]=str(listTime[5])
    
    df.to_excel(str(timepath)+'/'+str(label)+'.xlsx', sheet_name=label, index=False)
    print("tiempo total de las grabaciones de "+ label+":",total_time)                         


def featureTHALENTO(list_path, kfold, audio_type, label):
    clases ="binaria"
    # 1. Loading data from json list    
    for k in range(0, kfold):
        tic = time.time()
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                file_name =item['path'].split("/")
                train_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])                
                train_labels.append(int(item['label']))
                
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                file_name =item['path'].split("/")
                test_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])
                test_labels.append(int(item['label']))                
        f.close()

        # 2. Load features: Train
        # Get the same features.pkl for binaryclass and for multiclass
        try:
            audio_type_pkl = audio_type.split('multi_')[1]
        except:
            audio_type_pkl = audio_type
        try:
            label_csv = label.split('_Nomiss')[1]
        except:
            label_csv = label        

        train_labels = np.array(train_labels)

        trainpath = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
        trainpath_csv = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)        
        
        i = 0
        train_features = []
        data = []            
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='smile.log',
        )
        for wav in train_files:
            print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
            name = os.path.basename(wav)[:-4]
            fle= os.path.basename(wav)
            path_wav =wav.split(name)[0]                                
            for r, d, f in os.walk(path_wav):                                    
                for file in f:
                    if(str(file) == str(fle)):                        
                        path = r + '/' + file
                        print(str(i+1) + '. append: ' + file)
                        data.append(path)
                        print('Processing: ... ', name+'_smile.csv')
                        feat_one = smile.process_file(path)                                    
                        feat_one.to_csv('data/features/' + str(label)+"/" + str(name) +'_smile.csv')
            i = i+1                          
        
        
        # read wav files and extract emobase features on that file
        print('Processing: ... ')
        feat = smile.process_files(data)            
        train_features.append(feat.to_numpy()[3:])
        
        print('Saving: ... Train: ' + audio_type_pkl+'_smile.csv')
        feat.to_csv(trainpath_csv+'_smile.csv')
        
        print('Train: ' + str(i))
        train_features = np.array(train_features)
        with open(trainpath, 'wb') as fid:
            pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
        fid.close()

        # Test
        test_labels = np.array(test_labels)
        testpath = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
        testpath_csv = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)
        
        i = 0
        test_features = []
        data = []
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='smile.log',
        )
        for wav in test_files:
            print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
            name = os.path.basename(wav)[:-4]
            path_wav =wav.split(name)[0]                                
            for r, d, f in os.walk(path_wav):                                    
                for file in f:
                    if(str(file) == str(fle)):                        
                        path = r + '/' + file
                        print(str(i+1) + '. append: ' + file)
                        data.append(path)
                        print('Processing: ... ', name+'_smile.csv')
                        feat_one = smile.process_file(path)                                    
                        feat_one.to_csv('data/features/' + str(label)+"/" + str(name) +'_smile.csv')
            i = i+1                   
        
            # read wav files and extract emobase features on that file
        print('Processing: ... ')
        feat = smile.process_files(data)            
        test_features.append(feat.to_numpy()[3:])
        
        print('Saving: ... Test: ' + audio_type_pkl+'_smile.csv')
        feat.to_csv(testpath_csv+'_smile.csv')
        
        print('Test: ' + str(i))
        test_features = np.array(test_features)
        with open(testpath, 'wb') as fid:
            pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
        fid.close()
