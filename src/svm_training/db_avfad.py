from ast import And
from asyncio.windows_events import NULL
from functools import total_ordering
from genericpath import isdir, isfile
from multiprocessing.dummy import Array
import sys, json, os, pickle, time
import datetime
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import pandas as pd
import numpy as np
from sklearn.svm import SVC
import csv
import opensmile
import mutagen
from mutagen.wave import WAVE
from svm_training import utils, db_avfad, db_voiced, db_thalento, db_saarbruecken, Load_metadata as db
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


def timeAVFAD(list_path, path_metadata, label):     
    df = pd.read_excel (path_metadata , sheet_name= label)
    df = df.assign(Time="0:00:00")   
    df = df.assign(allTime="0:00:00") 
    total_time= datetime.timedelta(hours=00,minutes=00,seconds=00)
    
    timepath = 'data/pathology/' + label
    if not os.path.exists(timepath):
        os.makedirs(timepath)
    listPathology=["Abnormalities of the Vocal Fold", "control", "Dysphonia", "INFLAMMATORY CONDITIONS OF THE LARYNX", "OTHER DISORDERS AFFECTING VOICE", "Spasmodic Dysphonia"]
    type_audio=["a", "u", "i", "phrase"]
    timeAbnomalie= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeControl= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeDysphonia= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeInflammatory= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeOther= datetime.timedelta(hours=00,minutes=00,seconds=00)
    timeSpasmodic= datetime.timedelta(hours=00,minutes=00,seconds=00)
    listTime=[timeAbnomalie, timeControl,timeDysphonia, timeInflammatory, timeOther, timeSpasmodic ]
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
                        if pathology == df["CMVD-I word class"][i]:
                            listTime[j]= listTime[j] + time                            
                            break
                        j=j+1
                    #print(file,' Total Duration: {}:{}:{}'.format(hours, mins, seconds))
                    # j=0
                    # for pathology in type_audio:
                    #     if pathology == "a":
                    #         listTime[j]= listTime[j] + time                            
                    #         break
                    #     j=j+1
                    
                                                                           
        
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
    
    df['allTime'][0]=str(total_time)
    df["timeAbnomalie"][0]=str(listTime[0])
    df["timeControl"][0]=str(listTime[1])
    df["timeDysphonia"][0]=str(listTime[2])
    df["timeInflammatory"][0]=str(listTime[3])
    df["timeOther"][0]=str(listTime[4])
    df["timeSpasmodic"][0]=str(listTime[5])
    
    df.to_excel(str(timepath)+'/'+str(label)+'.xlsx', sheet_name=label, index=False)
    print("tiempo total de las grabaciones de "+ label+":",total_time)   

def featureAVFAD(list_path, kfold, audio_type, label): #reviar
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
                print(trainlist)
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
                                feat_one.to_csv(camino +"/" + str(name) +'_smile.csv')
                    i = i+1                          
                
                
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
                                feat_one.to_csv(camino+"/" + str(name) +'_smile.csv')
                    i = i+1                   
                
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

def feature_m_AVFAD(list_path, kfold, audio_type, label): #reviar
    clases ="Multiclass"    
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
                print(trainlist)
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
                                feat_one.to_csv(camino +"/" + str(name) +'_smile.csv')
                    i = i+1                          
                
                
                # # read wav files and extract emobase features on that file
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
                                feat_one.to_csv(camino+"/" + str(name) +'_smile.csv')
                    i = i+1                   
                
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
        

def svmAVFAD(list_path,kfold, audio_type, label):
    clases ="binario"
    #general=["male","female", 'both']
    general=['both']    
    #grabacion=["phrase","vowels", "a", "i", "u"]
    grabacion=["phrase"]
     # 1. Loading data from json list    
    ker = 'poly'
    d = 1
    c = 1
    
    for w in general:
            qq=len(grabacion)-1
            while qq >=0:
                    respath = 'data/result/' + label+"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                    if not os.path.exists(respath):
                        os.makedirs(respath)
        
                    result_log = str(respath)+'/results_' + label + '_' + clases + '_' + audio_type + '_' + ker + str(d) + 'c' + str(c) + '.log'
                    f = open(result_log, 'w+')
                    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
                    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
                    f.close()
                    
                    score = np.zeros((13, kfold))
                    score_oracle = np.zeros((13, kfold))
                    # 1. Loading data from json list
                    for k in range(0, kfold):
                        tic = time.time()
                        label_files =[]
                        train_files = []
                        train_labels = []
                        trainlist = list_path +"/"+ clases +"/"+ w+"/"+ w+'_'+grabacion[qq] + '/train_' + clases + '_' + grabacion[qq] + '_meta_data_fold' + str(k + 1) + '.json'
                        #trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
                        with open(trainlist, 'r') as f:
                            data = json.load(f)
                            for item in data['labels']:                
                                label_files.append(item)   
                                
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

                        # trainpath = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                        # if os.path.exists(trainpath) and cambia == 'viejo':
                        #     with open(trainpath, 'rb') as fid:
                        #         train_features = pickle.load(fid)
                        #     fid.close()
                        #     print('Fold ' + str(k + 1) + ' Train: ' + str(train_features.shape))
                        # else:
                        i = 0
                        train_features = []
                        for wav in train_files:                            
                            name = os.path.basename(wav)[:-4]
                            camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                            feat = pd.read_csv(camino + '/' + name + '_smile.csv').to_numpy()[0]
                            #feat = pd.read_csv('data/features/' + label_csv + '/' + name + '_smile.csv').to_numpy()[0]
                            train_features.append(feat[3:])
                            i = i + 1
                        
                        print(': Fold ' + str(k + 1) +'Train: ' + str(i))
                        train_features = np.array(train_features)
                        # with open(trainpath, 'wb') as fid:
                        #     pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
                        # fid.close()

                        train_features, trainmean, trainstd = utils.zscore(train_features)
                        # Test
                        test_labels = np.array(test_labels)
                        # testpath = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
                        # if os.path.exists(testpath) and cambia == 'viejo':
                        #     with open(testpath, 'rb') as fid:
                        #         test_features = pickle.load(fid)
                        #     print('Fold ' + str(k + 1) + ' Test: ' + str(test_features.shape))
                        #     fid.close()
                        # else:
                        i = 0
                        test_features = []
                        for wav in test_files:                          
                            name = os.path.basename(wav)[:-4]
                            camino= 'data/features/' + label +"/"+  clases +"/"+ w+"/"+ w+'_'+grabacion[qq]
                            feat = pd.read_csv(camino + '/' + name + '_smile.csv').to_numpy()[0]
                            #feat = pd.read_csv('data/features/' + label_csv + '/' + name + '_smile.csv').to_numpy()[0]
                            test_features.append(feat[3:])
                            i = i + 1
                        print(': Fold ' + str(k + 1) +'Test: ' + str(i))                        
                        test_features = np.array(test_features)
                        # with open(testpath, 'wb') as fid:
                        #     pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
                        # fid.close()
                        test_features = utils.zscore(test_features, trainmean, trainstd)

                        # 3. Train SVM classifier
                        counter = Counter(train_labels)
                        print('Norm: %i, Path: %i\n' % (counter[0], counter[1]))

                        clf = SVC(C=c, kernel=ker, degree=d, probability=True)
                        clf.fit(train_features, train_labels)

                        # 4. Testing
                         #     # 4. Testing
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
                                k + 1, toc - tic, score[0, k], score[1, k], score[2, k], score[3, k], score[4, k], score[5, k],
                                score[6, k],
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

def svm_m_AVFAD(list_path,kfold, audio_type, label):
    clases ="Multiclass"
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
            resumen=False
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
                clf.fit(train_features, train_labels)

                # 4. Testing
                out = clf.predict(test_features)
                out_oracle = clf.predict(train_features)

                #score[:, k] = utils.compute_score(clf, test_labels, out, test_features)
                score = utils.compute_score_multiclass(test_labels, out, label_code, resumen, label)        
                #score_oracle[:, k] = utils.compute_score(clf, train_labels, out_oracle, train_features)
                score_oracle = utils.compute_score_multiclass(train_labels, out_oracle, label_code, resumen, label)
                
                
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
        #print("%s %s" % (train, test))
    return dict_fold

def salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, dict_clases, name_base, grabacion, genero):
    fold_train = {}; fold_test = {}; tipo_clases = {}
    
    #tipo = "phrase"  if grabacion.replace('_both','') == "phrase" else  grabacion.replace('_both','')
    tipo = grabacion
    fold_train = {"labels": dict_clases, "meta_data": []}
    fold_test = {"labels": dict_clases, "meta_data": []}

    train = fold_train['meta_data']
    test = fold_test['meta_data']
    vowels=[]
    index = 0
    for i in muestras_train:
        spk = dict_info_signal[i]['spk']
        label = list_clases_train[index]

        index = index + 1
        #gender = 'hombres'
        #if dict_info_signal[i]['gender'] == 'w':
        #   gender = 'mujeres'
        
        if genero=="male":
            if tipo == "phrase"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "a"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "i"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "vowels"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                ii = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                uu = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
                train.append(ii)
                train.append(uu)                
        
        if genero=="female":
            if tipo == "phrase" and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "a" and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "i"and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "vowels" and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                ii = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                uu = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
                train.append(ii)
                train.append(uu)                
                    
        if genero=="both":
            if tipo == "phrase":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "a":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "u":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "i":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
            if tipo == "vowels":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                ii = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                uu = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                train.append(aa)
                train.append(ii)
                train.append(uu)                    
        
    fold_train['meta_data'] = train

    index = 0
    for i in muestras_test:
        spk = dict_info_signal[i]['spk']
        label = list_clases_test[index]
        index = index + 1
        
        if genero=="male":
            if tipo == "phrase"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "a"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "i"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "vowels"and dict_info_signal[i]['gender'] == 'M':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                ii = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                uu = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
                test.append(ii)
                test.append(uu)                
        
        if genero=="female":
            if tipo == "phrase" and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "a" and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "u"and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "i"and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "vowels" and dict_info_signal[i]['gender'] == 'F':
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                ii = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                uu = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
                test.append(ii)
                test.append(uu)                
                    
        if genero=="both":
            if tipo == "phrase":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + tipo + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "a":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "u":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "i":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
            if tipo == "vowels":
                aa = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "002" + '.wav', 'label': label, 'speaker': spk}
                ii = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "001" + '.wav', 'label': label, 'speaker': spk}
                uu = {'path': 'data/audio/' + name_base + '/'+ str(i) +'/' + str(i) + "003" + '.wav', 'label': label, 'speaker': spk}
                test.append(aa)
                test.append(ii)
                test.append(uu) 
    fold_test['meta_data'] = test

    return fold_train, fold_test

def kford():
    name_base="AVFAD"
    dict_info_signal = db.main('data/lst/' + name_base + '/' + name_base + '_metadata.xlsx', name_base)
    b_list_muestras, b_list_clases, b_list_grupos, b_dict_clases = binaria_Cross_validation(dict_info_signal);
    m_list_muestras, m_list_clases, m_list_grupos, m_dict_clases = multi_Cross_validation(dict_info_signal);
    
    b_fold = StratifiedGroupKFold_G(b_list_muestras, b_list_clases, b_list_grupos, 5)
    m_fold = StratifiedGroupKFold_G(m_list_muestras, m_list_clases, m_list_grupos, 5)
    
    #intercambio
    pp=0
    while( pp<5):
        fold = 'fold'+str(pp)
        
        test_end= int(len(np.array(b_fold[fold]['test'])) / 2)
        for item in range(0, test_end, 3):
            try:                   
                aux= b_fold[fold]['test'][item]
                b_fold[fold]['test'][item] = b_fold[fold]['train'][item]
                b_fold[fold]['train'][item]=aux
            except:
                pass
        
        for item in range(0, test_end,5):            
            foldnext = 'fold'+str(pp+1)
            try:               
                if pp < 4:
                    aux= b_fold[fold]['test'][item]
                    b_fold[fold]['test'][item] = b_fold[fold]['train'][item]
                    b_fold[fold]['train'][item]=aux
                else:
                    aux= b_fold[fold]['test'][item]
                    b_fold[fold]['test'][item] = b_fold['fold0']['train'][item]
                    b_fold['fold0']['train'][item]=aux
                    
            except:
                pass 
        
        end= int(len(np.array(b_fold[fold]['train'])) / 2)
        for item in range(0, end,2):                        
            try:               
                aux= b_fold[fold]['train'][item]
                b_fold[fold]['train'][item] = b_fold[fold]['test'][item]
                b_fold[fold]['test'][item]=aux                
            except:
                pass            
        for item in range(0, end,5):            
            foldnext = 'fold'+str(pp+1)
            try:               
                if pp < 4:
                    aux= b_fold[fold]['train'][item]
                    b_fold[fold]['train'][item] = b_fold[foldnext]['test'][item]
                    b_fold[foldnext]['test'][item]=aux
                else:
                    aux= b_fold[fold]['train'][item]
                    b_fold[fold]['train'][item] = b_fold['fold0']['test'][item]
                    b_fold['fold0']['test'][item]=aux
                    
            except:
                pass            
        
                   
        
        pp+=1
        
    clases=["binario", "Multiclass"]    
    # general=["male","female", 'both']
    # grabacion=["phrase","vowels", "a", "i", "u"]    
        
    general=['both']
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
                ind_test = np.array(m_fold[i]['test'])
                muestras_train = np.array(m_list_muestras)[ind_train]
                list_clases_train = np.array(m_list_clases)[ind_train]
                muestras_test = np.array(m_list_muestras)[ind_test]
                list_clases_test = np.array(m_list_clases)[ind_test]        
                
                
                [fold_train, fold_test] = salva_fold_binaria(muestras_train, list_clases_train, muestras_test, list_clases_test, dict_info_signal, m_dict_clases, name_base, grabacion[j], k)
                if not os.path.exists(camino):
                    os.makedirs(camino + '/')

                with open(camino + '/' + 'train_' + clases[1] + '_' + grabacion[j] + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
                    json.dump(fold_train, file, indent=6)
                file.close()
                with open(camino + '/' + 'test_' + clases[1] + '_' + grabacion[j] + '_meta_data_fold' + str(ind) + '.json', 'w') as file:
                    json.dump(fold_test, file, indent=6)
                file.close()                
                j = j - 1            
        ind = ind + 1
    print("lolo")