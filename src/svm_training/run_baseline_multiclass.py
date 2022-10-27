import sys, json, os, pickle, time
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from svm_training import utils
# from utils import compute_score_multiclass, zscore
from collections import Counter
import matplotlib.pyplot as plt

def main(list_path, kfold, audio_type, resumen = True):

    ker='poly'
    d=1
    c=1
    label = os.path.basename(list_path)

    result_log = 'results_'+label+'_multiclases_' +audio_type+'_'+ker+str(d)+'c'+str(c)+'.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

    respath = 'data/result/'+label
    if not os.path.exists(respath): os.mkdir(respath)
    
    # 1. Loading data from json list
    dic_result_oracle = {}; dic_result = {}
    for k in range(0,kfold):
        tic = time.time()
        train_files = [] 
        train_labels = [] 
        trainlist = list_path + '/train_multiclases_' + audio_type + '_meta_data_fold' + str(k+1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            #c = 0
            for item in data['meta_data']:
                train_files.append(item['path'])
                train_labels.append(item['label'])
                #c +=1
                #print(str(c) + ' ' + item['path'] + ' ' + str(item['label']))
        f.close()

        test_files = [] 
        test_labels = [] 
        testlist = list_path + '/test_multiclases_' + audio_type + '_meta_data_fold' + str(k+1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            #c = 0
            for item in data['meta_data']:
                test_files.append(item['path'])
                test_labels.append(item['label'])
                #c +=1
                #print(str(c) + ' ' + item['path'] + ' ' + str(item['label']))
        f.close()


        # 2. Load features: Train
        #audio_type_pkl = audio_type.split('multi_')[1]
        train_labels = np.array(train_labels)
        trainpath = 'data/features/'+label+'/train_multiclases_'+ audio_type +'_fold'+str(k+1)+'.pkl'
        if os.path.exists(trainpath):
            with open(trainpath,'rb') as fid:
                train_features = pickle.load(fid)
                print('Fold '+ str(k+1) +' Train: ' + str(train_features.shape))
        else:
            i=0
            train_features = []
            for wav in train_files:
                print(str(i) + ': Fold ' + str(k+1) + ': ' + wav)
                name = os.path.basename(wav)[:-4]
                feat = pd.read_csv('data/features/'+label+'/'+name+'_smile.csv').to_numpy()[0]
                train_features.append(feat[3:])
                i=i+1
            print('Train: ' + str(i))          
            train_features = np.array(train_features)
            with open(trainpath,'wb') as fid:
                pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
        train_features, trainmean, trainstd = utils.zscore(train_features)
        # Test
        test_labels = np.array(test_labels)
        testpath = 'data/features/'+label+'/test_multiclases_'+audio_type+'_fold'+str(k+1)+'.pkl'
        if os.path.exists(testpath):
            with open(testpath,'rb') as fid:
                test_features = pickle.load(fid)
                print('Fold '+ str(k+1) + ' Test: ' + str(test_features.shape))
        else:
            i=0
            test_features = []
            for wav in test_files:
                print(str(i) + ': Fold ' + str(k+1) + ': ' + wav)
                name = os.path.basename(wav)[:-4]          
                feat = pd.read_csv('data/features/'+label+'/'+name+'_smile.csv').to_numpy()[0]
                test_features.append(feat[3:])
                i=i+1
            print('Test: ' + str(i))
            test_features = np.array(test_features) 
            with open(testpath,'wb') as fid:
                pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)            
        test_features = utils.zscore(test_features, trainmean, trainstd)


        # 3. Train SVM classifier
        counter = Counter(train_labels)
        labels = data['labels'].keys()
        sizes = []
        for i in labels:
            sizes.append(counter[data['labels'][i]])
        '''
        plt.subplot(121)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Train')  
        '''
        counter = Counter(test_labels)
        sizes = []
        for i in labels:
            sizes.append(counter[data['labels'][i]])
        '''
        plt.subplot(122)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Test')  
       # plt.show()
        '''
        clf = SVC(C=c,kernel=ker,degree=d)
        clf.fit(train_features, train_labels)

        # 4. Testing
        out = clf.predict(test_features)
        out_oracle = clf.predict(train_features)

        score = utils.compute_score_multiclass(test_labels, out, data, resumen)
        score_oracle = utils.compute_score_multiclass(train_labels, out_oracle, data, resumen)
        
        lbl = {}
        for i in labels:
            lbl[data['labels'][i]] = i
        with open (respath+'/output_'+audio_type+'_fold'+str(k+1)+'_'+ker+'d'+str(d)+'c'+str(c)+'.log', 'w') as f:
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



        

    





    




#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('audio_type (SVD): multi_a_n, multi_aiu, multi_phrases')
        print('audio_type (AVFAD): multi_aiu, multi_phrases, multi_read, multi_spontaneous')
        print('Usage: run_baseline.py list_path kfold audio_type')
        print('Example: python run_baseline.py data/lst 5 phrase_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
