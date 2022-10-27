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

def svmTHALENTO(list_path,kfold, audio_type, label):
    clases = "multiclases"
    ker = 'poly'
    d = 1
    c = 1
    resumen=True
    respath = 'data/result/' + label
    if not os.path.exists(respath):
        os.mkdir(respath)
        
    result_log = str(respath)+'/results_' + label + '_' + clases + '_' + audio_type + '_' + ker + str(d) + 'c' + str(c) + '.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()
    
    score = np.zeros((13, kfold))
    score_oracle = np.zeros((13, kfold))
    dic_result_oracle = {}; dic_result = {}
    # 1. Loading data from json list
    for k in range(0, kfold):
        tic = time.time()
        label_files =[]
        label_code ={}
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
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
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
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

        trainpath = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
        # if os.path.exists(trainpath) and cambia == 'viejo':
        #     with open(trainpath, 'rb') as fid:
        #         train_features = pickle.load(fid)
        #     fid.close()
        #     print('Fold ' + str(k + 1) + ' Train: ' + str(train_features.shape))
        # else:
        i = 0
        train_features = []
        for wav in train_files:
            print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
            name = os.path.basename(wav)[:-4]
            feat = pd.read_csv('data/features/' + label_csv + '/' + name + '_smile.csv').to_numpy()[0]
            train_features.append(feat[3:])
            i = i + 1
        print('Train: ' + str(i))
        train_features = np.array(train_features)
        with open(trainpath, 'wb') as fid:
            pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
        fid.close()

        train_features, trainmean, trainstd = utils.zscore(train_features)
        # Test
        test_labels = np.array(test_labels)
        testpath = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
        # if os.path.exists(testpath) and cambia == 'viejo':
        #     with open(testpath, 'rb') as fid:
        #         test_features = pickle.load(fid)
        #     print('Fold ' + str(k + 1) + ' Test: ' + str(test_features.shape))
        #     fid.close()
        # else:
        i = 0
        test_features = []
        for wav in test_files:
            print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
            name = os.path.basename(wav)[:-4]
            feat = pd.read_csv('data/features/' + label_csv + '/' + name + '_smile.csv').to_numpy()[0]
            test_features.append(feat[3:])
            i = i + 1
        print('Test: ' + str(i))
        test_features = np.array(test_features)
        with open(testpath, 'wb') as fid:
            pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
        fid.close()
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
        score = utils.compute_score_multiclass(test_labels, out, label_files, resumen)        
        #score_oracle[:, k] = utils.compute_score(clf, train_labels, out_oracle, train_features)
        score_oracle = utils.compute_score_multiclass(train_labels, out_oracle, label_files, resumen)
        
        
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

