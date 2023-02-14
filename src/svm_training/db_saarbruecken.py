from ast import And
from asyncio.windows_events import NULL
from functools import total_ordering
from genericpath import isdir, isfile
from multiprocessing.dummy import Array
import sys, json, os, pickle, time
import datetime
from tokenize import group
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, train_test_split
import random
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import opensmile
import mutagen
from mutagen.wave import WAVE
#from utils import compute_score, zscore
from svm_training import utils, clustering, db_avfad, db_voiced, db_thalento, db_saarbruecken, Load_metadata as db
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
    clases ="binario"       
    general=["male","female", 'both']              
    grabacion=["phrase","vowels", "a", "i", "u"]
    
     # 1. Loading data from json list    
    for k in range(0, kfold):
        for w in general:
            j=len(grabacion)-1
            while j >=0:
                tic = time.time()
                train_files = []
                train_labels = []
                #trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                trainlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[j] + '/train_' + clases + '_' + grabacion[j] + '_meta_data_fold' + str(k + 1) + '.json'
                
                with open(trainlist, 'r') as f:
                    data = json.load(f)
                    for item in data['meta_data']:
                        file_name =item['path'].split("/")
                        train_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])                
                        train_labels.append(int(item['label']))
                        
                f.close()

                test_files = []
                test_labels = []
                #testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                testlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[j] + '/test_' + clases + '_' + grabacion[j] + '_meta_data_fold' + str(k + 1) + '.json'
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
                camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[j]
                if not os.path.exists(camino):
                    os.makedirs(camino + '/')


                #trainpath = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                #trainpath_csv = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)        
                trainpath = camino  + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                trainpath_csv = camino  + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)        
                
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
                    name = os.path.basename(wav)[:-4]
                    fle= os.path.basename(wav)
                    path_wav =wav.split(name)[0]                                
                    for r, d, f in os.walk(path_wav):                                    
                        for file in f:
                            if(str(file) == str(fle)):                        
                                path = r + '/' + file                                
                                data.append(path)                                
                                feat_one = smile.process_file(path)                                    
                                feat_one.to_csv(camino +"/" + str(name) +'_smile.csv')                    
                    i = i+1                          
                
                ##Esto es para crear csv y pkl generico por fold 
                # read wav files and extract emobase features on that file
                # print('Processing: ... ')
                # feat = smile.process_files(data)            
                # train_features.append(feat.to_numpy()[3:])
                
                # print('Saving: ... Train: ' + audio_type_pkl+'_smile.csv')
                # feat.to_csv(trainpath_csv+'_smile.csv')
                
                # print('Train: ' + str(i))
                # train_features = np.array(train_features)
                # with open(trainpath, 'wb') as fid:
                #     pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
                # fid.close()

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
                    name = os.path.basename(wav)[:-4]
                    path_wav =wav.split(name)[0]                                
                    for r, d, f in os.walk(path_wav):                                    
                        for file in f:
                            if(str(file) == str(fle)):                        
                                path = r + '/' + file                                
                                data.append(path)                                
                                feat_one = smile.process_file(path)                                    
                                feat_one.to_csv(camino+"/" + str(name) +'_smile.csv')                                
                    
                    i = i+1                   
                
                ##Esto es para crear csv y pkl generico por fold 
                # read wav files and extract emobase features on that file
                # print('Processing: ... ')
                # feat = smile.process_files(data)            
                # test_features.append(feat.to_numpy()[3:])
                
                # print('Saving: ... Test: ' + audio_type_pkl+'_smile.csv')
                # feat.to_csv(testpath_csv+'_smile.csv')
                
                # print('Test: ' + str(i))
                # test_features = np.array(test_features)
                # with open(testpath, 'wb') as fid:
                #     pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
                # fid.close()
                
                
                j=j-1
        print('Fold ' + str(k + 1) +'--Saving: ... ', camino)
    
def feature_m_Saarbruecken(list_path, kfold, audio_type, label): #reviar
    clases ="Multiclass"    
    #general=["male","female", 'both']     
    #grabacion=["phrase","vowels", "a", "i", "u"]
    general=['both']     
    grabacion=["phrase"]
    
     # 1. Loading data from json list    
    for k in range(0, kfold):
        for w in general:
            j=len(grabacion)-1
            while j >=0:
                tic = time.time()
                train_files = []
                train_labels = []
                #trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                trainlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[j] + '/train_' + clases + '_' + grabacion[j] + '_meta_data_fold' + str(k + 1) + '.json'
                
                with open(trainlist, 'r') as f:
                    data = json.load(f)
                    for item in data['meta_data']:
                        file_name =item['path'].split("/")
                        train_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])                
                        train_labels.append(int(item['label']))
                        
                f.close()

                test_files = []
                test_labels = []
                #testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                testlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[j] + '/test_' + clases + '_' + grabacion[j] + '_meta_data_fold' + str(k + 1) + '.json'
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
                camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[j]
                if not os.path.exists(camino):
                    os.makedirs(camino + '/')


                #trainpath = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                #trainpath_csv = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)        
                trainpath = camino  + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                trainpath_csv = camino  + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)        
                
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
                    name = os.path.basename(wav)[:-4]
                    fle= os.path.basename(wav)
                    path_wav =wav.split(name)[0]                                
                    for r, d, f in os.walk(path_wav):                                    
                        for file in f:
                            if(str(file) == str(fle)):                        
                                path = r + '/' + file                                
                                data.append(path)                                
                                feat_one = smile.process_file(path)                                    
                                feat_one.to_csv(camino +"/" + str(name) +'_smile.csv')                    
                    i = i+1                          
                
                ##Esto es para crear csv y pkl generico por fold 
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
                #testpath = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                #testpath_csv = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)
                testpath = camino  + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                testpath_csv = camino  + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)        
                
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
                    name = os.path.basename(wav)[:-4]
                    path_wav =wav.split(name)[0]                                
                    for r, d, f in os.walk(path_wav):                                    
                        for file in f:
                            if(str(file) == str(fle)):                        
                                path = r + '/' + file                                
                                data.append(path)                                
                                feat_one = smile.process_file(path)                                    
                                feat_one.to_csv(camino+"/" + str(name) +'_smile.csv')                    
                    i = i+1                   
                
                ##Esto es para crear csv y pkl generico por fold 
                # # read wav files and extract emobase features on that file
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
                
                j=j-1
        print('Fold ' + str(k + 1) +'--Saving: ... ', camino)

         
def svmSaarbruecken(list_path,kfold, audio_type, label):
    clases ="binario"
    ker = 'poly'
    d = 1
    c = 1
    #general=["male","female", 'both']       
    general=['both']          
    #grabacion=["phrase","vowels", "a", "i", "u"]
    grabacion=["phrase"]
    for w in general:
            qq=len(grabacion)-1
            while qq >=0:
                    respath = 'data/result/' + label+"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]                    
                    if not os.path.exists(respath):
                        os.makedirs(respath)
                    
                    result_log = str(respath) +'/results_' + label + '_' + clases + '_' + audio_type + '_' + ker + str(d) + 'c' + str(c) + '.log'
                    f = open(result_log, 'w+')
                    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
                    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
                    f.close()
                    
                    score = np.zeros((13, kfold))
                    score_oracle = np.zeros((13, kfold))
                    # 1. Loading data from json list
                    dic_result_oracle = {}; dic_result = {}
                    for k in range(0, kfold):
                        tic = time.time()
                        label_files =[]
                        label_code ={}
                        train_files = []
                        train_labels = []
                        trainlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[qq] + '/train_' + clases + '_' + grabacion[qq] + '_meta_data_fold' + str(k + 1) + '.json'
                        #trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                        with open(trainlist, 'r') as f:
                            data = json.load(f)                        
                            for item in data['labels']:                
                                label_files.append(item)                                  
                                label_code[int(data['labels'][item])] = item
                                
                            for item in data['meta_data']:
                                train_files.append(item['path'])                
                                train_labels.append(int(item['label']))   
                                        
                        f.close()

                        test_files = []
                        test_labels = []  
                        testlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[qq] + '/test_' + clases + '_' + grabacion[qq] + '_meta_data_fold' + str(k + 1) + '.json'      
                        #testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                        with open(testlist, 'r') as f:
                            data = json.load(f)            
                            for item in data['meta_data']:
                                test_files.append(item['path'])                
                                test_labels.append(int(item['label']))
                                
                        f.close()

                        # 2. Load features: Train                   

                        if not os.path.exists('data/features/' + label): os.mkdir('data/features/' + label)

                        train_labels = np.array(train_labels)
                        i = 0
                        train_features = []
                        for wav in train_files:                            
                            name = os.path.basename(wav)[:-4]
                            camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                            feat = pd.read_csv(camino + '/' + name + '_smile.csv').to_numpy()[0]
                            train_features.append(feat[3:])
                            i = i + 1
                        print(': Fold ' + str(k + 1) + 'Train: ' + str(i))                        
                        train_features = np.array(train_features)

                        train_features, trainmean, trainstd = utils.zscore(train_features)
                        # Test        
                        test_labels = np.array(test_labels)        
                        i = 0
                        test_features = []
                        for wav in test_files:                            
                            name = os.path.basename(wav)[:-4]
                            camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                            feat = pd.read_csv(camino + '/' + name + '_smile.csv').to_numpy()[0]
                            test_features.append(feat[3:])
                            i = i + 1                        
                        print(': Fold ' + str(k + 1) + 'Test: ' + str(i))
                        test_features = np.array(test_features)        
                        test_features = utils.zscore(test_features, trainmean, trainstd)

                        # 3. Train SVM classifier
                        counter = Counter(train_labels)
                        print('Norm: %i, Path: %i\n' % (counter[0], counter[1]))

                        clf = SVC(C=c, kernel=ker, degree=d, probability=True)
                        clf.fit(train_features, train_labels)

                        # 4. Testing
                        out = clf.predict(test_features)
                        out_oracle = clf.predict(train_features)
                        
                        score[:, k] = utils.compute_score(clf, test_labels, out, test_features)
                        with open(respath + '/output_' + audio_type + '_fold' + str(k + 1) + '_' + ker + 'd' + str(d) + 'c' + str(
                                c) + '.log', 'w') as f:
                            lbl = ['NORM', 'PATH']
                            for j in range(0, len(test_files)):
                                f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[out[j]]))

                        score_oracle[:, k] = utils.compute_score(clf, train_labels, out_oracle, train_features)
                        with open(
                                respath + '/output_oracle_' + audio_type + '_fold' + str(k + 1) + '_' + ker + 'd' + str(d) + 'c' + str(
                                        c) + '.log', 'w') as f:
                            lbl = ['NORM', 'PATH']
                            for j in range(0, len(train_files)):
                                f.write(
                                    '%s %s %s\n' % (os.path.basename(train_files[j])[:-4], lbl[train_labels[j]], lbl[out_oracle[j]]))

                        toc = time.time()
                        f = open(result_log, 'a')
                        f.write(
                            'Oracle Fold%i (%.2fsec): Acc=%0.4f, AccNorm=%0.2f, AccPath=%0.2f, UAR=%0.4f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.4f, EER=%0.4f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f \n' %
                            (k + 1, toc - tic, score_oracle[0, k], score_oracle[1, k], score_oracle[2, k], score_oracle[3, k],
                            score_oracle[4, k], score_oracle[5, k], score_oracle[6, k], score_oracle[7, k], score_oracle[8, k],
                            score_oracle[9, k], score_oracle[10, k], score_oracle[11, k], score_oracle[12, k]))
                        f.close()
                        toc = time.time()
                        f = open(result_log, 'a')
                        f.write(
                            'Test Fold%i (%.2fsec): Acc=%0.4f, AccNorm=%0.2f, AccPath=%0.2f, UAR=%0.4f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.4f, EER=%0.4f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f \n\n' %
                            (
                            k + 1, toc - tic, score[0, k], score[1, k], score[2, k], score[3, k], score[4, k], score[5, k], score[6, k],
                            score[7, k], score[8, k], score[9, k], score[10, k], score[11, k], score[12, k]))
                        f.close()

                    f = open(result_log, 'a')
                    f.write(
                        'TOTAL Oracle: Acc=%0.4f, AccNorm=%0.2f, AccPath=%0.2f, UAR=%0.4f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.2f, EER=%0.4f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f \n' %
                        (np.mean(score_oracle[0, :]), np.mean(score_oracle[1, :]), np.mean(score_oracle[2, :]),
                        np.mean(score_oracle[3, :]), np.mean(score_oracle[4, :]), np.mean(score_oracle[5, :]),
                        np.mean(score_oracle[6, :]), np.mean(score_oracle[7, :]), np.mean(score_oracle[8, :]),
                        np.mean(score_oracle[9, :]), np.mean(score_oracle[10, :]), np.mean(score_oracle[11, :]),
                        np.mean(score_oracle[12, :])))
                    f.close()

                    f = open(result_log, 'a')
                    f.write(
                        'TOTAL Test: Acc=%0.4f, AccNorm=%0.2f, AccPath=%0.2f, UAR=%0.4f, F1Score=%0.2f, Recall=%0.2f, Precision=%0.2f, AUC=%0.4f, EER=%0.4f, TP=%0.2f, TN=%0.2f, FP=%0.2f, FN=%0.2f \n\n' %
                        (np.mean(score[0, :]), np.mean(score[1, :]), np.mean(score[2, :]), np.mean(score[3, :]), np.mean(score[4, :]),
                        np.mean(score[5, :]),
                        np.mean(score[6, :]), np.mean(score[7, :]), np.mean(score[8, :]), np.mean(score[9, :]), np.mean(score[10, :]),
                        np.mean(score[11, :]), np.mean(score[12, :])))
                    f.close()
                    
                    qq-=1

def svm_m_Saarbruecken(list_path,kfold, audio_type, label):
    clases ="Multiclass"
    ker = 'poly'
    d = 2
    c = 1
    # general=["male","female", 'both']       
    general=['both']          
    # grabacion=["phrase","vowels", "a", "i", "u"]
    grabacion=["phrase"]
    for w in general:
        qq=len(grabacion)-1
        while qq >=0:
            resumen=False
            respath = 'data/result/' + label+"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]                    
            if not os.path.exists(respath):
                os.makedirs(respath)
            
            result_log = str(respath) +'/results_' + label + '_' + clases + '_' + audio_type + '_' + ker + str(d) + 'c' + str(c) + '.log'
            
            f = open(result_log, 'w+')
            f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
            f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
            #f.write('MLPClassifier: random_state=1,max_iter=300 \n')            
            #f.write('RandomForestClassifier:max_depth=2, random_state=0 \n')            
            f.close()
            
            score = np.zeros((13, kfold))
            score_oracle = np.zeros((13, kfold))
            # 1. Loading data from json list
            dic_result_oracle = {}; dic_result = {}
            for k in range(0, kfold):
                tic = time.time()
                label_files =[]
                label_code ={}
                train_files = []
                train_labels = []
                trainlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[qq] + '/train_' + clases + '_' + grabacion[qq] + '_meta_data_fold' + str(k + 1) + '.json'
                #trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                with open(trainlist, 'r') as f:
                    data = json.load(f)                        
                    for item in data['labels']:                
                        label_files.append(item)                                  
                        label_code[int(data['labels'][item])] = item
                        
                    for item in data['meta_data']:
                        train_files.append(item['path'])                
                        train_labels.append(int(item['label']))   
                                
                f.close()

                test_files = []
                test_labels = []
                testlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[qq] + '/test_' + clases + '_' + grabacion[qq] + '_meta_data_fold' + str(k + 1) + '.json'              
                #testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                with open(testlist, 'r') as f:
                    data = json.load(f)            
                    for item in data['meta_data']:
                        test_files.append(item['path'])                
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

                if not os.path.exists('data/features/' + label): os.mkdir('data/features/' + label)

                train_labels = np.array(train_labels)
                i = 0
                train_features = []
                for wav in train_files:                        
                    name = os.path.basename(wav)[:-4]
                    camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                    feat = pd.read_csv(camino + '/' + name + '_smile.csv').to_numpy()[0]                        
                    train_features.append(feat[3:])
                    i = i + 1                    
                print('Fold ' + str(k + 1) + 'Train: ' + str(i))
                train_features = np.array(train_features)        

                train_features, trainmean, trainstd = utils.zscore(train_features)
                # Test        
                test_labels = np.array(test_labels)        
                i = 0
                test_features = []
                for wav in test_files:                        
                    name = os.path.basename(wav)[:-4]
                    camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                    feat = pd.read_csv(camino + '/' + name + '_smile.csv').to_numpy()[0]
                    test_features.append(feat[3:])
                    i = i + 1
                print('Fold ' + str(k + 1) + 'Test: ' + str(i))
                test_features = np.array(test_features)
                test_features = utils.zscore(test_features, trainmean, trainstd)

                # 3. Train SVM classifier
                # counter = Counter(train_labels)
                # print('Norm: %i, Path: %i\n' % (counter[0], counter[1]))

                clf = SVC(C=c, kernel=ker, degree=d, probability=True)
                #clf = MLPClassifier(random_state=1, max_iter=300)
                #clf = RandomForestClassifier(max_depth=2, random_state=0)
                clf.fit(train_features, train_labels)

                # 4. Testing
                out = clf.predict(test_features)
                out_oracle = clf.predict(train_features)

                
                score = utils.compute_score_multiclass(test_labels, out, label_files, resumen, label)        
                score_oracle = utils.compute_score_multiclass(train_labels, out_oracle, label_files, resumen, label)
                
                
                lbl = label_code           
                with open(respath + '/output_' + audio_type + '_fold' + str(k + 1) + '_' + ker + 'd' + str(d) + 'c' + str(c) + '.log', 'w') as f:        
                    for j in range(0,len(test_files)):                                
                        f.write('%s %s %s\n' % (os.path.basename(test_files[j])[:-4], lbl[test_labels[j]], lbl[out[j]]))
                
                with open (respath+'/output_oracle_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
                    for j in range(0,len(train_files)):
                        f.write('%s %s %s\n' % (os.path.basename(train_files[j])[:-4], lbl[train_labels[j]], lbl[out_oracle[j]]))

                toc = time.time()
                f = open(result_log, 'a')
                if resumen:
                    f.write('\nOracle Fold%i (%.2fsec)          precision    recall  f1-score   support' % (k + 1, toc - tic))
                    
                    dic_result_oracle['accuracy' + str(k)] = score_oracle['accuracy']
                    dic_result_oracle['macro avg' + str(k)] = score_oracle['macro avg']
                    dic_result_oracle['weighted avg' + str(k)] = score_oracle['weighted avg']

                    aux1 = str(round(score_oracle['macro avg']['precision'], 2))
                    aux2 = str(round(score_oracle['macro avg']['recall'], 2))
                    aux3 = str(round(score_oracle['macro avg']['f1-score'], 2))
                    aux4 = str(score_oracle['macro avg']['support'])

                    f.write('\nmacro avg                          ' + aux1 + '        ' + aux2 + '     ' + aux3 + '      ' + aux4)

                    aux1 = str(round(score_oracle['weighted avg']['precision'], 2))
                    aux2 = str(round(score_oracle['weighted avg']['recall'], 2))
                    aux3 = str(round(score_oracle['weighted avg']['f1-score'], 2))
                    f.write('\nweighted avg                       ' + aux1 + '        ' + aux2 + '     ' + aux3 + '      ' + aux4)
                    f.write(
                        '\naccuracy                 ' + str(round(score_oracle['accuracy'], 2)))

                    f.write('\n');  f.write('\n');

                    f.write('\nTest Fold%i (%.2fsec)            precision    recall  f1-score   support' % (k + 1, toc - tic))

                    dic_result['accuracy' + str(k)] = 0
                    if 'accuracy' in score:
                        dic_result['accuracy' + str(k)] = score['accuracy']
                    dic_result['macro avg' + str(k)] = score['macro avg']
                    dic_result['weighted avg' + str(k)] = score['weighted avg']

                    aux1 = str(round(score['macro avg']['precision'], 2))
                    aux2 = str(round(score['macro avg']['recall'], 2))
                    aux3 = str(round(score['macro avg']['f1-score'], 2))
                    aux4 = str(score['macro avg']['support'])

                    f.write('\nmacro avg                          ' + aux1 + '        ' + aux2 + '     ' + aux3 + '      ' + aux4)

                    aux1 = str(round(score['weighted avg']['precision'], 2))
                    aux2 = str(round(score['weighted avg']['recall'], 2))
                    aux3 = str(round(score['weighted avg']['f1-score'], 2))
                    f.write('\nweighted avg                       ' + aux1 + '        ' + aux2 + '     ' + aux3 + '      ' + aux4)
                    f.write('\naccuracy                 ' + str(round(dic_result['accuracy' + str(k)], 2)))

                    f.write('\n'); f.write('\n');

                else:
                    f.write('\nOracle Fold%i (%.2fsec)' % (k + 1, toc - tic))
                    f.write(score_oracle)
                    f.write('\nTest Fold%i (%.2fsec)' % (k + 1, toc-tic))
                    f.write(score)
                f.close()
                
            if resumen:
                accuracy_oracle = 0; macro_precision_oracle = 0; macro_recall_oracle = 0; macro_f1score_oracle = 0;
                support_oracle = 0; weighted_precision_oracle = 0; weighted_recall_oracle = 0; weighted_f1score_oracle = 0;
                accuracy = 0; macro_precision = 0; macro_recall = 0; macro_f1score = 0; support = 0
                weighted_precision = 0; weighted_recall = 0; weighted_f1score = 0;
                for k in range(0, kfold):
                    accuracy_oracle = accuracy_oracle + dic_result_oracle['accuracy' + str(k)]
                    aux = dic_result_oracle['macro avg' + str(k)]
                    macro_precision_oracle = macro_precision_oracle + aux['precision']
                    macro_recall_oracle = macro_recall_oracle + aux['recall']
                    macro_f1score_oracle = macro_f1score_oracle + aux['f1-score']
                    support_oracle = support_oracle + aux['support']
                    aux = dic_result_oracle['weighted avg' + str(k)]
                    weighted_precision_oracle = weighted_precision_oracle + aux['precision']
                    weighted_recall_oracle = weighted_recall_oracle + aux['recall']
                    weighted_f1score_oracle = weighted_f1score_oracle + aux['f1-score']

                    accuracy = accuracy + dic_result['accuracy' + str(k)]
                    aux = dic_result['macro avg' + str(k)]
                    macro_precision = macro_precision + aux['precision']
                    macro_recall = macro_recall + aux['recall']
                    macro_f1score = macro_f1score + aux['f1-score']
                    support = support + aux['support']
                    aux = dic_result['weighted avg' + str(k)]
                    weighted_precision = weighted_precision + aux['precision']
                    weighted_recall = weighted_recall + aux['recall']
                    weighted_f1score = weighted_f1score + aux['f1-score']

                f = open(result_log, 'a')
                f.write('\n');  f.write('\n');  f.write('\n'); f.write('\n'); f.write('\n')
                f.write('\nOracle 5 Fold average            precision    recall  f1-score   support')
                aux1 = str(round(macro_precision_oracle / kfold,2))
                aux2 = str(round(macro_recall_oracle / kfold,2))
                aux3 = str(round(macro_f1score_oracle / kfold,2))
                aux4 = str(round(support_oracle / kfold,2))
                aux = aux1 + '        ' + aux2 + '    ' + aux3 + '       ' + aux4
                f.write('\nmacro avg                          ' + aux)
                aux1 = str(round(weighted_precision_oracle / kfold, 2))
                aux2 = str(round(weighted_recall_oracle / kfold, 2))
                aux3 = str(round(weighted_f1score_oracle / kfold, 2))
                aux4 = str(round(support_oracle / kfold, 2))
                aux = aux1 + '        ' + aux2 + '    ' + aux3 + '       ' + aux4
                f.write('\nweighted avg                       ' + aux)
                f.write('\naccuracy                 ' + str(round(accuracy_oracle / kfold, 2)))

                f.write('\n'); f.write('\n'); f.write('\n');

                f.write('\nTest 5 Fold average              precision    recall  f1-score   support')
                aux1 = str(round(macro_precision / kfold, 2))
                aux2 = str(round(macro_recall / kfold, 2))
                aux3 = str(round(macro_f1score / kfold, 2))
                aux4 = str(round(support / kfold, 2))
                aux = aux1 + '        ' + aux2 + '     ' + aux3 + '       ' + aux4
                f.write('\nmacro avg                          ' + aux)
                aux1 = str(round(weighted_precision / kfold, 2))
                aux2 = str(round(weighted_recall / kfold, 2))
                aux3 = str(round(weighted_f1score / kfold, 2))
                aux4 = str(round(support / kfold, 2))
                aux = aux1 + '        ' + aux2 + '      ' + aux3 + '        ' + aux4
                f.write('\nweighted avg                       ' + aux)
                f.write('\naccuracy                 ' + str(round(accuracy / kfold, 2)))
                f.close()
            qq -=1      
    
def clustering_m_Saarbruecken(list_path,kfold, audio_type, label):
    data_path="data/features/Saarbruecken/Multiclass/both/both_phrase"
    df = pd.read_excel("data/lst/Saarbruecken/Saarbruecken_metadata.xlsx", sheet_name='Saarbruecken')
    dict_info_signal = {}
    i = 0
    data = []
    label=[]
    y_label=[]
    files=[]
    for ind, row in df.iterrows():                
        p= "PATH" if df['PATHOLOGY'][ind]=='p' else "NORM"
        g= "hombres" if df['GENDER'][ind]=='m' else "mujeres"
        path=  "data/audio/Saarbruecken"+"/"+p+"/"+g+"/"+str(row[0])            
        ##nuevas pathology
        dict_info_signal[row[0]] = {'spk': row[0],'Path': path,'pathology': str(row[13]).strip().upper(), 'group': row[16]}
        file=str(row[0])+'-phrase_smile.csv'
        feat = pd.read_csv(data_path+'/'+file).to_numpy()[0]
        data.append(feat[3:])
        label.append(row[16])
        y_label.append(str(row[13]).strip().upper())
        y_label.append(str(row[13]).strip().upper())
        files.append(path)
        # if i ==10:
        #     break;
        # i+=1
            
    
    
    
    # 3. Clustering
    clustering.cluster(data, label, y_label, files )

                
            
           
            

def binaria_Cross_validation(sesion):
    list_muestras = []
    list_clases = []
    list_grupos = []
    dict_clases = {"HEALTHY": '0', "PATH": '1'}
    for j in sesion:
        list_muestras.append(j)
        list_grupos.append(sesion[j]['group'])
        if sesion[j]['group'] == 0:
            list_clases.append('0')
        else:
            list_clases.append('1')        
            
    return list_muestras, list_clases, list_grupos, dict_clases

def multi_Cross_validation(sesion):
    list_muestras = []
    list_clases = []
    list_grupos = []
    dict_clases = {}
    for j in sesion:
        list_muestras.append(j)
        list_grupos.append(str(sesion[j]['group']))        
        
        dict_clases[sesion[j]['pathology']] = str(sesion[j]['group'])
        list_clases.append(str(sesion[j]['group']))
       
    
    return list_muestras, list_clases, list_grupos, dict_clases

def StratifiedGroupKFold_G(X, y, groups, kfold = 5):
    sgkf = StratifiedGroupKFold(n_splits = kfold)    
    dict_fold = {};
    i = 0
    for train, test in sgkf.split(X, y, groups=groups):
        dict_fold['fold' + str(i)] = {'train': train, 'test': test}
        i = i + 1
        # print("%s %s" % (train, test))
    return dict_fold

def GroupKFold_G(X, y, groups, kfold = 5):
    gkf = GroupKFold(n_splits = kfold)
    dict_fold = {}; 
    i = 0
    for train, test in gkf.split(X, y, groups=groups):
        dict_fold['fold' + str(i)] = {'train': train, 'test': test}
        i = i + 1
        #print("%s %s" % (train, test))
    return dict_fold

def spliter_custom(list_muestras, list_clases, nfold):        
    split= int(len(list_muestras) / nfold)
    X= np.array(random.sample(list_muestras, split))
    y= np.full(len(X), list_clases[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # X_train, X_test, y_train, y_test = train_test_split(list_muestras, list_clases, test_size=0.20)
    return X_train, X_test, y_train, y_test

def StratifiedGroupKFold_M(kfold = 5, sign= NULL):    
    dict_fold = {};
    label=[]
    value=[]
    for i in sign:
        if not sign[i]['pathology'] in label:
            label.append(sign[i]['pathology'])      
            value.append(sign[i]['group'])
     
    trains0=[];trains1=[];trains2=[];trains3=[];trains4=[]
    y_trains0=[];y_trains1=[];y_trains2=[];y_trains3=[];y_trains4=[]
    tests0=[];tests1=[];tests2=[];tests3=[];tests4=[]   
    y_tests0=[];tests1=[];tests2=[];tests3=[];tests4=[]   
    index = 0     
    for w in label:        
        list_muestras = []
        list_clases = []
        list_grupos = []
        dict_clases = {}    
        for j in sign:            
            if sign[j]['pathology'] == w:
                list_muestras.append(j)
                list_grupos.append(str(sign[j]['group']))
                dict_clases[sign[j]['pathology']] = str(sign[j]['group'])
                list_clases.append(str(sign[j]['group']))
        
                                
        for i in range(0,kfold):
            if index == 0:
                if i == 0:
                    trains0,tests0, y_trains0, y_tests0=spliter_custom(list_muestras, list_clases, kfold)                    
                if i == 1:
                    trains1,tests1, y_trains1, y_tests1 =spliter_custom(list_muestras, list_clases, kfold)                    
                if i == 2:
                    trains2,tests2, y_trains2, y_tests2 =spliter_custom(list_muestras, list_clases, kfold)                    
                if i == 3:
                    trains3,tests3, y_trains3, y_tests3 =spliter_custom(list_muestras, list_clases, kfold)
                if i == 4:
                    trains4,tests4, y_trains4, y_tests4 =spliter_custom(list_muestras, list_clases, kfold)
            else:
                if i == 0:
                    train, test, y_train, y_test = spliter_custom(list_muestras, list_clases, kfold)
                    trains0=np.concatenate((trains0,train)) 
                    tests0= np.concatenate((tests0, test))
                    y_trains0=np.concatenate((y_trains0,y_train)) 
                    y_tests0= np.concatenate((y_tests0, y_test))
                if i == 1:
                    train, test, y_train, y_test = spliter_custom(list_muestras, list_clases, kfold)
                    trains1= np.concatenate((trains1, train))
                    tests1= np.concatenate((tests1, test))
                    y_trains1=np.concatenate((y_trains1,y_train)) 
                    y_tests1= np.concatenate((y_tests1, y_test))
                if i == 2:
                    train, test, y_train, y_test = spliter_custom(list_muestras, list_clases, kfold)
                    trains2= np.concatenate((trains2, train))
                    tests2= np.concatenate((tests2, test))
                    y_trains2=np.concatenate((y_trains2,y_train)) 
                    y_tests2= np.concatenate((y_tests2, y_test))
                if i == 3:
                    train, test, y_train, y_test = spliter_custom(list_muestras, list_clases, kfold)
                    trains3= np.concatenate((trains3, train))
                    tests3= np.concatenate((tests3, test))
                    y_trains3=np.concatenate((y_trains3,y_train)) 
                    y_tests3= np.concatenate((y_tests3, y_test))
                if i == 4:
                    train, test, y_train, y_test = spliter_custom(list_muestras, list_clases, kfold)
                    trains4= np.concatenate((trains4, train))
                    tests4= np.concatenate((tests4, test))
                    y_trains4=np.concatenate((y_trains4,y_train)) 
                    y_tests4= np.concatenate((y_tests4, y_test))
            i = i + 1
            # print("%s %s" % (train, test))
    
        index +=1
    dict_fold['fold0' ] = {'train': trains0 , 'test': tests0, 'y_train': y_trains0 , 'y_test': y_tests0  }
    dict_fold['fold1' ] = {'train': trains1 , 'test': tests1, 'y_train': y_trains1 , 'y_test': y_tests1  }
    dict_fold['fold2' ] = {'train': trains2 , 'test': tests2, 'y_train': y_trains2 , 'y_test': y_tests2  }
    dict_fold['fold3' ] = {'train': trains3 , 'test': tests3, 'y_train': y_trains3 , 'y_test': y_tests3  }
    dict_fold['fold4' ] = {'train': trains4 , 'test': tests4, 'y_train': y_trains4 , 'y_test': y_tests4  }
  
    return dict_fold

def salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, dict_clases, name_base, grabacion, genero):
    fold_train = {}; fold_test = {}; tipo_clases = {}
    
    #tipo = "phrase"  if grabacion.replace('_both','') == "phrase" else  grabacion.replace('_both','')
    tipo = grabacion
    fold_train = {"labels": {}, "meta_data": []}
    fold_test = {"labels": {}, "meta_data": []}

    train = fold_train['meta_data']
    test = fold_test['meta_data']
    classes_train={}
    classes_test={}      
    index = 0
    for i in muestras_train:
        spk = dict_info_signal[i]['spk']
        label = list_clases_train[index]
        index = index + 1        
        path=  dict_info_signal[i]['Path']  
        classes_train[dict_info_signal[i]['pathology']]=str(dict_info_signal[i]['group'])
        
        if genero=="male":
            if tipo == "phrase"and dict_info_signal[i]['gender'] == 'm':
                aa = {'path': path+ '-phrase.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "a"and dict_info_signal[i]['gender'] == 'm':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'm':                
                aa0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)                
            if tipo == "i"and dict_info_signal[i]['gender'] == 'm':                
                aa0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
                
            if tipo == "vowels"and dict_info_signal[i]['gender'] == 'm':                
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)                
                uu0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                uu1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                uu2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                uu3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                train.append(uu0)
                train.append(uu1)
                train.append(uu2)
                train.append(uu3)
                ii0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                ii1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                ii2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                ii3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                train.append(ii0)
                train.append(ii1)
                train.append(ii2)
                train.append(ii3)
        
        if genero=="female":
            if tipo == "phrase" and dict_info_signal[i]['gender'] == 'w':
                aa = {'path': path+ '-phrase.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "a" and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3) 
            if tipo == "i"and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
            if tipo == "vowels" and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)                
                uu0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                uu1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                uu2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                uu3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                train.append(uu0)
                train.append(uu1)
                train.append(uu2)
                train.append(uu3)
                ii0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                ii1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                ii2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                ii3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                train.append(ii0)
                train.append(ii1)
                train.append(ii2)
                train.append(ii3)                
                    
        if genero=="both":
            if tipo == "phrase":
                aa = {'path': path+ '-phrase.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "a":
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
            if tipo == "u":
                aa0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
            if tipo == "i":
                aa0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)
            if tipo == "vowels":
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                train.append(aa0)
                train.append(aa1)
                train.append(aa2)
                train.append(aa3)                
                uu0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                uu1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                uu2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                uu3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                train.append(uu0)
                train.append(uu1)
                train.append(uu2)
                train.append(uu3)
                ii0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                ii1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                ii2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                ii3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                train.append(ii0)
                train.append(ii1)
                train.append(ii2)
                train.append(ii3)                    
    classes_train = json.dumps(classes_train, sort_keys=True)        
    fold_train['labels'] =  json.loads(classes_train)
    fold_train['meta_data'] = train

    index = 0
    for i in muestras_test:
        spk = dict_info_signal[i]['spk']
        label = list_clases_test[index]
        index = index + 1
        path=  dict_info_signal[i]['Path']
        
        classes_test[dict_info_signal[i]['pathology']]=str(dict_info_signal[i]['group'])
        
        if genero=="male":
            if tipo == "phrase"and dict_info_signal[i]['gender'] == 'm':
                aa = {'path': path+ '-phrase.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "a"and dict_info_signal[i]['gender'] == 'm':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'm':
                aa0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "i"and dict_info_signal[i]['gender'] == 'm':
                aa0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "vowels"and dict_info_signal[i]['gender'] == 'm':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)                
                uu0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                uu1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                uu2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                uu3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                test.append(uu0)
                test.append(uu1)
                test.append(uu2)
                test.append(uu3)
                ii0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                ii1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                ii2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                ii3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                test.append(ii0)
                test.append(ii1)
                test.append(ii2)
                test.append(ii3)                
        
        if genero=="female":
            if tipo == "phrase" and dict_info_signal[i]['gender'] == 'w':
                aa = {'path': path+ '-phrase.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "a" and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "i"and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "vowels" and dict_info_signal[i]['gender'] == 'w':
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)                
                uu0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                uu1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                uu2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                uu3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                test.append(uu0)
                test.append(uu1)
                test.append(uu2)
                test.append(uu3)
                ii0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                ii1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                ii2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                ii3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                test.append(ii0)
                test.append(ii1)
                test.append(ii2)
                test.append(ii3)               
                    
        if genero=="both":
            if tipo == "phrase":
                aa = {'path': path+ '-phrase.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "a":
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "u":
                aa0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "i":
                aa0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)
            if tipo == "vowels":
                aa0 = {'path': path+ '-a_h.wav', 'label': label, 'speaker': spk}
                aa1 = {'path': path+ '-a_l.wav', 'label': label, 'speaker': spk}
                aa2 = {'path': path+ '-a_lhl.wav', 'label': label, 'speaker': spk}
                aa3 = {'path': path+ '-a_n.wav', 'label': label, 'speaker': spk}
                test.append(aa0)
                test.append(aa1)
                test.append(aa2)
                test.append(aa3)                
                uu0 = {'path': path+ '-u_h.wav', 'label': label, 'speaker': spk}
                uu1 = {'path': path+ '-u_l.wav', 'label': label, 'speaker': spk}
                uu2 = {'path': path+ '-u_lhl.wav', 'label': label, 'speaker': spk}
                uu3 = {'path': path+ '-u_n.wav', 'label': label, 'speaker': spk}
                test.append(uu0)
                test.append(uu1)
                test.append(uu2)
                test.append(uu3)
                ii0 = {'path': path+ '-i_h.wav', 'label': label, 'speaker': spk}
                ii1 = {'path': path+ '-i_l.wav', 'label': label, 'speaker': spk}
                ii2 = {'path': path+ '-i_lhl.wav', 'label': label, 'speaker': spk}
                ii3 = {'path': path+ '-i_n.wav', 'label': label, 'speaker': spk}
                test.append(ii0)
                test.append(ii1)
                test.append(ii2)
                test.append(ii3) 
    
    classes_test = json.dumps(classes_test, sort_keys=True)        
    fold_test['labels'] = json.loads(classes_test)
    fold_test['meta_data'] = test
    return fold_train, fold_test

def GroupPathology_G(sign, m_list_muestras, m_list_clases, m_list_grupos, nfold):    
    label=[]
    index={}
    for i in sign:
        if not sign[i]['pathology'] in label:
            label.append(sign[i]['pathology'])                   
        
    count= np.zeros(len(label), dtype=int)
    for i in sign:        
        count[label.index(sign[i]['pathology'])]+=1

    print("Label", '===>' ,"Cantidad")
    for x in range(0, len(label)):
        print(label[x], '===>' ,count[x])
        
    dict_fold = {};
    i = 0    
    return dict_fold


def CountPathology_Fold(sign, name):    
    label=[]
    num=[]   
    met=sign['meta_data']     
    for i in sign['labels']:
        if not i in label:
            label.append(i)
            num.append(sign['labels'][i])                   
     
    count= np.zeros(len(label), dtype=int)
    for idx, x in enumerate(num):
        for i in range(0,len(met)):
            if x == met[i]['label']:
                count[idx]+=1

    print(name)
    for x in range(0, len(label)):
        print(label[x], '===>' ,count[x])
        
    
def kford():
    name_base="Saarbruecken"
    ### Aqui se hace la lista en dependencia del typo audio (phrase, vowels, a, i, u)
    dict_info_signal = db.main('data/lst/' + name_base + '/' + name_base + '_metadata.xlsx', name_base)
    b_list_muestras, b_list_clases, b_list_grupos, b_dict_clases = binaria_Cross_validation(dict_info_signal);
    m_list_muestras, m_list_clases, m_list_grupos, m_dict_clases = multi_Cross_validation(dict_info_signal);
    
    b_fold = StratifiedGroupKFold_G(b_list_muestras, b_list_clases, b_list_grupos, 5)
    m_fold = StratifiedGroupKFold_M(5, dict_info_signal)
    # m_fold = GroupKFold_G(m_list_muestras, m_list_clases, m_list_grupos, 5)
    
    
    
    # #intercambio
    # pp=0
    # while( pp<5):
    #     fold = 'fold'+str(pp)
        
    #     test_end= int(len(np.array(b_fold[fold]['test'])) / 2)
    #     for item in range(0, test_end, 3):
    #         try:                   
    #             aux= b_fold[fold]['test'][item]
    #             b_fold[fold]['test'][item] = b_fold[fold]['train'][item]
    #             b_fold[fold]['train'][item]=aux
    #         except:
    #             pass
        
    #     for item in range(0, test_end,5):            
    #         foldnext = 'fold'+str(pp+1)
    #         try:               
    #             if pp < 4:
    #                 aux= b_fold[fold]['test'][item]
    #                 b_fold[fold]['test'][item] = b_fold[fold]['train'][item]
    #                 b_fold[fold]['train'][item]=aux
    #             else:
    #                 aux= b_fold[fold]['test'][item]
    #                 b_fold[fold]['test'][item] = b_fold['fold0']['train'][item]
    #                 b_fold['fold0']['train'][item]=aux
                    
    #         except:
    #             pass 
        
    #     end= int(len(np.array(b_fold[fold]['train'])) / 2)
    #     for item in range(0, end,2):                        
    #         try:               
    #             aux= b_fold[fold]['train'][item]
    #             b_fold[fold]['train'][item] = b_fold[fold]['test'][item]
    #             b_fold[fold]['test'][item]=aux                
    #         except:
    #             pass            
    #     for item in range(0, end,5):            
    #         foldnext = 'fold'+str(pp+1)
    #         try:               
    #             if pp < 4:
    #                 aux= b_fold[fold]['train'][item]
    #                 b_fold[fold]['train'][item] = b_fold[foldnext]['test'][item]
    #                 b_fold[foldnext]['test'][item]=aux
    #             else:
    #                 aux= b_fold[fold]['train'][item]
    #                 b_fold[fold]['train'][item] = b_fold['fold0']['test'][item]
    #                 b_fold['fold0']['test'][item]=aux
                    
    #         except:
    #             pass            
        
                   
        
    #     pp+=1
    clases=["binario", "Multiclass"]    
    general=['both']        
    # general=["male","female", 'both']    
    # grabacion=["phrase","vowels", "a", "i", "u"]
    grabacion=["phrase"]
    ind = 1; camino = 'data/lst/' + name_base
    for i in b_fold:
        for k in general:
            j=len(grabacion)-1
            while j >=0:
                ## Binario                
                # camino = 'data/lst/' + name_base+"/"+ clases[0]+"/"+k+"/"+k+"_"+grabacion[j]
                # ind_train = np.array(b_fold[i]['train'])
                # ind_test = np.array(b_fold[i]['test'])
                # muestras_train = np.array(b_list_muestras)[ind_train]
                # list_clases_train = np.array(b_list_clases)[ind_train]
                # muestras_test = np.array(b_list_muestras)[ind_test]
                # list_clases_test = np.array(b_list_clases)[ind_test]        
                
                
                # [fold_train, fold_test] = salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, b_dict_clases, name_base, grabacion[j], k)
                # if not os.path.exists(camino):
                #     os.makedirs(camino + '/')

                # with open(camino + '/' + 'train_' + clases[0] + '_' + grabacion[j] + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
                #     json.dump(fold_train, file, indent=6)
                # file.close()
                # with open(camino + '/' + 'test_' + clases[0] + '_' + grabacion[j] + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
                #     json.dump(fold_test, file, indent=6)
                # file.close() 
                
                ## Multiclass
                camino = 'data/lst/' + name_base+"/"+ clases[1]+"/"+k+"/"+k+"_"+grabacion[j]
                ind_train = np.array(m_fold[i]['train'])
                ind_ytrain = np.array(m_fold[i]['y_train'])
                ind_test = np.array(m_fold[i]['test'])
                ind_ytest = np.array(m_fold[i]['y_test'])
                
                muestras_train = [a for a in m_list_muestras if a in ind_train]
                list_clases_train = ind_ytrain
                # muestras_train = np.array(m_list_muestras)[ind_train]
                # list_clases_train = np.array(m_list_clases)[ind_train]
                muestras_test = [a for a in m_list_muestras if a in ind_test]                
                list_clases_test = ind_ytest                
                # muestras_test = np.array(m_list_muestras)[ind_test]
                # list_clases_test = np.array(m_list_clases)[ind_test]        
                
                
                [fold_train, fold_test] = salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, m_dict_clases, name_base, grabacion[j], k)
                if not os.path.exists(camino):
                    os.makedirs(camino + '/')
                
                with open(camino + '/' + 'train_' + clases[1] + '_' + grabacion[j] + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
                    json.dump(fold_train, file, indent=6)
                    # CountPathology_Fold(fold_train, 'train fold' + str(ind))
                file.close()
                with open(camino + '/' + 'test_' + clases[1] + '_' + grabacion[j] + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
                    json.dump(fold_test, file, indent=6)
                    # CountPathology_Fold(fold_test, 'test fold' + str(ind))
                file.close()                
                j = j - 1            
        ind = ind + 1
    print("lolo")



