from asyncio.windows_events import NULL
import sys, json, os, pickle, time

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


def main_with_thalento(list_path, list_path_test, kfold, audio_type, audio_type_test, cambia='viejo', clases='binaria'):
    ker = 'poly'
    d = 1
    c = 1
    label = os.path.basename(list_path)
    label_test = os.path.basename(list_path_test)

    result_log = 'results_' + label_test + '_' + clases + '_' + audio_type_test + '_' + ker + str(d) + 'c' + str(
        c) + '.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label_test, kfold, audio_type_test))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

    respath = 'data/result/' + label_test
    if not os.path.exists(respath):
        os.mkdir(respath)

    score = np.zeros((13, kfold))
    score_oracle = np.zeros((13, kfold))
    # 1. Loading data from json list
    for k in range(0, kfold):
        tic = time.time()
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                train_files.append(item['path'])
                if item['label'] == '0':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path_test + '/test_' + clases + '_' + audio_type_test + '_meta_data_fold' + str(k + 1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                test_files.append(item['path'])
                if item['label'] == '0':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
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
        if os.path.exists(trainpath) and cambia == 'viejo':
            with open(trainpath, 'rb') as fid:
                train_features = pickle.load(fid)
            fid.close()
            print('Fold ' + str(k + 1) + ' Train: ' + str(train_features.shape))
        else:
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

        # 2. Load features: Test
        # Get the same features.pkl for binaryclass and for multiclass
        try:
            audio_type_pkl_test = audio_type.split('multi_')[1]
        except:
            audio_type_pkl_test = audio_type_test
        try:
            label_csv_test = label_test.split('_Nomiss')[1]
        except:
            label_csv_test = label_test

        test_labels = np.array(test_labels)
        testpath = 'data/features/' + label_test + '/test_' + clases + '_' + audio_type_pkl_test + '_fold' + str(
            k + 1) + '.pkl'
        if os.path.exists(testpath) and cambia == 'viejo':
            with open(testpath, 'rb') as fid:
                test_features = pickle.load(fid)
            print('Fold ' + str(k + 1) + ' Test: ' + str(test_features.shape))
            fid.close()
        else:
            i = 0
            test_features = []
            for wav in test_files:
                print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
                name = os.path.basename(wav)[:-4]
                feat = pd.read_csv('data/features/' + label_csv_test + '/' + name + '_smile.csv').to_numpy()[0]
                test_features.append(feat[3:])
                i = i + 1
            print('Test: ' + str(i))
            test_features = np.array(test_features)
            with open(testpath, 'wb') as fid:
                pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
            fid.close()
        test_features = utils.zscore(test_features, trainmean, trainstd)

        # 3. Train SVM classifier
        counter = Counter(train_labels)
        print('Norm: %i, Path: %i\n' % (counter[0], counter[1]))

        clf = SVC(C=c, kernel=ker, degree=d, probability=True)
        clf.fit(train_features, train_labels)

        # pickle.dump(clf, open("SVM_Model.pkl", 'wb'))
        # pickle.dump(trainmean, open("Stat_Mean.pkl", 'wb'))
        # pickle.dump(trainstd, open("Stat_STD.pkl", 'wb'))


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


def main(list_path=NULL, kfold=5, audio_type=NULL, cambia='viejo', clases='binaria'):
    ker = 'poly'
    d = 1
    c = 1
    label = os.path.basename(list_path)

    result_log = 'results_' + label + '_' + clases + '_' + audio_type + '_' + ker + str(d) + 'c' + str(c) + '.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

    respath = 'data/result/' + label
    if not os.path.exists(respath):
        os.mkdir(respath)

    score = np.zeros((13, kfold))
    score_oracle = np.zeros((13, kfold))
    # 1. Loading data from json list
    for k in range(0, kfold):
        tic = time.time()
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                train_files.append(item['path'])
                if item['label'] == '0':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                test_files.append(item['path'])
                if item['label'] == '0':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
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
        if os.path.exists(trainpath) and cambia == 'viejo':
            with open(trainpath, 'rb') as fid:
                train_features = pickle.load(fid)
            fid.close()
            print('Fold ' + str(k + 1) + ' Train: ' + str(train_features.shape))
        else:
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
        if os.path.exists(testpath) and cambia == 'viejo':
            with open(testpath, 'rb') as fid:
                test_features = pickle.load(fid)
            print('Fold ' + str(k + 1) + ' Test: ' + str(test_features.shape))
            fid.close()
        else:
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


def crea_paquetes(list_path, kfold, audio_type, cambia='viejo', clases='binaria'):
    label = os.path.basename(list_path)

    respath = 'data/result/' + label
    if not os.path.exists(respath):
        #os.mkdir(respath)
        os.makedirs(respath, exist_ok=True)
    # 1. Loading data from json list
    for k in range(0, kfold):
        tic = time.time()
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                train_files.append(item['path'])
                if item['label'] == '0':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                test_files.append(item['path'])
                if item['label'] == '0':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
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
        if os.path.exists(trainpath) and cambia == 'viejo':
            with open(trainpath, 'rb') as fid:
                train_features = pickle.load(fid)
            fid.close()
            print('Fold ' + str(k + 1) + ' Train: ' + str(train_features.shape))
        else:
            i = 0
            train_features = []
            for wav in train_files:
                print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
                name = os.path.basename(wav)[:-4]

                # with open('data/features/' + label_csv + '/' + name + '_praat.csv', 'r') as file:
                #     reader = csv.reader(file)
                #     for row in reader:
                #         row_float = [float(x) for x in row]
                #         train_features.append(row_float)

                feat = pd.read_csv('data/features/' + label_csv + '/' + name + '_smile.csv').to_numpy()[0]
                train_features.append(feat[3:])
                i = i + 1
            print('Train: ' + str(i))
            train_features = np.array(train_features)
            with open(trainpath, 'wb') as fid:
                pickle.dump(train_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
            fid.close()

        # Test
        test_labels = np.array(test_labels)
        testpath = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1) + '.pkl'
        if os.path.exists(testpath) and cambia == 'viejo':
            with open(testpath, 'rb') as fid:
                test_features = pickle.load(fid)
            print('Fold ' + str(k + 1) + ' Test: ' + str(test_features.shape))
            fid.close()
        else:
            i = 0
            test_features = []
            for wav in test_files:
                print(str(i) + ': Fold ' + str(k + 1) + ': ' + wav)
                name = os.path.basename(wav)[:-4]

                # with open('data/features/' + label_csv + '/' + name + '_praat.csv', 'r') as file:
                #     reader = csv.reader(file)
                #     for row in reader:
                #         row_float = [float(x) for x in row]
                #         test_features.append(row_float)
                #
                feat = pd.read_csv('data/features/' + label_csv + '/' + name + '_smile.csv').to_numpy()[0]
                test_features.append(feat[3:])
                i = i + 1
            print('Test: ' + str(i))
            test_features = np.array(test_features)
            with open(testpath, 'wb') as fid:
                pickle.dump(test_features, fid, protocol=pickle.HIGHEST_PROTOCOL)
            fid.close()

def feature_smile(list_path, kfold, audio_type, cambia='viejo', clases='binaria'):
    label = os.path.basename(list_path)

    respath = 'data/result/' + label
    if not os.path.exists(respath):
        #os.mkdir(respath)
        os.makedirs(respath, exist_ok=True)
        
    respath = 'data/features/' + label
    if not os.path.exists(respath):
        #os.mkdir(respath)
        os.makedirs(respath, exist_ok=True)
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
                if item['label'] == '0':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                file_name =item['path'].split("/")
                test_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])
                
                if item['label'] == '0':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
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
        if os.path.exists(trainpath) and cambia == 'viejo':
            with open(trainpath, 'rb') as fid:
                train_features = pickle.load(fid)
            fid.close()
            print('Fold ' + str(k + 1) + ' Train: ' + str(train_features.shape))
        else:
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
                path_wav =wav.split(name)[0]                                
                for r, d, f in os.walk(path_wav):                                    
                  for file in f:
                    if "_LECTURA" in file:
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
        #testpath_csv = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold' + str(k + 1)
        testpath_csv = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl
        if os.path.exists(testpath) and cambia == 'viejo':
            with open(testpath, 'rb') as fid:
                test_features = pickle.load(fid)
            print('Fold ' + str(k + 1) + ' Test: ' + str(test_features.shape))
            fid.close()
        else:
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
                    if "_LECTURA" in file:
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

### Describir metodo
def svm_binario(list_path, kfold,audio_type, clases='binaria'):
    ker = 'poly'
    d = 1
    c = 1
    label = os.path.basename(list_path)

    result_log = 'results_' + label + '_' + clases + '_' + audio_type + '_' + ker + str(d) + 'c' + str(c) + '.log'
    f = open(result_log, 'w+')
    f.write('Results Data:%s Features:Compare2016 %ifold, %s\n' % (label, kfold, audio_type))
    f.write('SVM Config: Kernel=%s, Degree=%i, C(tol)=%.2f \n' % (ker, d, c))
    f.close()

    respath = 'data/result/' + label
    if not os.path.exists(respath):
        os.mkdir(respath)

    score = np.zeros((13, kfold))
    score_oracle = np.zeros((13, kfold))
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

    
    trainpath_csv = 'data/features/' + label + '/train_' + clases + '_' + audio_type_pkl + '_fold'
    testpath_csv = 'data/features/' + label + '/test_' + clases + '_' + audio_type_pkl + '_fold'
    
    print(testpath_csv)
    nfold = kfold
    while nfold > 0: 
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(nfold) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                train_files.append(item['path'])
                if item['label'] == '0':
                    train_labels.append("NORM")
                else:
                    train_labels.append("PATH")
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(nfold) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            for item in data['meta_data']:
                test_files.append(item['path'])
                if item['label'] == '0':
                    test_labels.append("NORM")
                else:
                    test_labels.append("PATH")
        f.close()
             
        X_train, y_train = utils.split_data(str(trainpath_csv)+str(nfold)+"_smile.csv", train_labels)
        X_test, y_test = utils.split_data(str(testpath_csv)+str(nfold)+"_smile.csv", test_labels)
        
        # Creación del modelo SVM
        # ==============================================================================      
        model=[]
        data_processing = {'list':[],'model':[],'type_list':[], 'Kerner':[] , 'value_C':[], 'value_degree':[], 'Score':[], 'UAR':[]}      
        modelo = SVC(kernel='poly')
        
        #length=range(1,64, 4) 
        length=range(1,2) 
        for c in length:            
                for deg in range(1,2):                  
                    modelo.C = c
                    modelo.degree = deg                
                    modelo.fit(X_train, y_train)
                    score= modelo.score(X_test, y_test) 
                    y_pred = modelo.predict(X_test)
                    #uar = balanced_accuracy_score(y_test, y_pred)             
                    
                    data_processing['list'].append("fold"+str(nfold)+"")
                    data_processing['model'].append("SVC")
                    data_processing['type_list'].append("Phrase")
                    data_processing['Kerner'].append("poly")
                    data_processing['value_C'].append(c)
                    data_processing['value_degree'].append(deg)
                    data_processing['Score'].append(score*100)
                    #data_processing['UAR'].append(uar*100)  
                      # 4. Testing
                    out = modelo.predict(X_test)
                    out_oracle = modelo.predict(X_train)
                    score[:, nfold] = utils.compute_score(modelo, test_labels, out, X_test)            
        
        exppath = 'data/xlsx/' + label
        if not os.path.exists(exppath):
            os.mkdir(exppath)
        #df = pd.DataFrame(data_processing, columns = ['list','model','type_list', 'Kerner' , 'value_C', 'value_degree', 'Score','UAR'])
        df = pd.DataFrame(data_processing, columns = ['list','model','type_list', 'Kerner' , 'value_C', 'value_degree', 'Score'])
        df.to_excel(str(exppath)+'/svm_binario_fold'+str(nfold)+'.xlsx', sheet_name='svm_binario', index=False)
        nfold -= 1 

# function to convert the information into 
# some readable format
def audio_duration(length):
    hours = length // 3600  # calculate in hours
    length %= 3600
    mins = length // 60  # calculate in minutes
    length %= 60
    seconds = length  # calculate in seconds
  
    return hours, mins, seconds  # returns the duration
  
       
def tiempo_total(list_path, kfold,audio_type, clases='binaria'):
    label = os.path.basename(list_path)
    for k in range(0, kfold):
        
        data_processing = {'list':[],'label':[],'time':[]}      
        
        train_files = []
        train_labels = []
        trainlist = list_path + '/train_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(trainlist, 'r') as f:
            data = json.load(f)
            i = 0
            for item in data['meta_data']:
                file_name =item['path'].split("/")
                train_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])
                if item['label'] == '0':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
                
                name = os.path.basename(train_files[i])[:-4]
                path_wav =train_files[i].split(name)[0]               
                
                for r, d, n in os.walk(path_wav):                                    
                  for file in n:
                    if "_LECTURA" in file:
                        path = r + '/' + file                                             
                        audio = WAVE(path)
                        audio_info = audio.info
                        length = int(audio_info.length)
                        hours, mins, seconds = audio_duration(length)
                        data_processing['list'].append(path)
                        data_processing['label'].append(item['label'])
                        data_processing['time'].append(str(hours)+":"+str(mins)+":"+str(seconds))
                        
                        
                i=i+1
        f.close()

        test_files = []
        test_labels = []
        testlist = list_path + '/test_' + clases + '_' + audio_type + '_meta_data_fold' + str(k + 1) + '.json'
        with open(testlist, 'r') as f:
            data = json.load(f)
            i = 0
            for item in data['meta_data']:
                file_name =item['path'].split("/")
                test_files.append(item['path'].split("-"+audio_type)[0]+"/"+ file_name[len(file_name)-1])
                
                if item['label'] == '0':
                    test_labels.append(0)
                else:
                    test_labels.append(1)
                
                name = os.path.basename(train_files[i])[:-4]
                path_wav =train_files[i].split(name)[0]               
                
                for r, d, n in os.walk(path_wav):                                    
                  for file in n:
                    if "_LECTURA" in file:
                        path = r + '/' + file
                        print(str(i+1) + '. append: ' + file)                        
                        audio = WAVE(path)
                        audio_info = audio.info
                        length = int(audio_info.length)
                        hours, mins, seconds = audio_duration(length)
                        data_processing['list'].append(path)
                        data_processing['label'].append(item['label'])
                        data_processing['time'].append(str(hours)+":"+str(mins)+":"+str(seconds))
                i=i+1
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
        
        timepath = 'data/time/' + label
        if not os.path.exists(timepath):
            os.makedirs(timepath)
        
        df = pd.DataFrame(data_processing, columns = ['list','label','time'])
        df.to_excel(str(timepath)+'/time_fold'+str(k)+'.xlsx', sheet_name='time_fold', index=False)
        
        
# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('audio_type (SVD): a_n, aiu, phrases, multi_a_n, multi_aiu, multi_phrases')
        print(
            'audio_type (AVFAD): aiu, phrases, read, spontaneous, multi_aiu, multi_phrases, multi_read, multi_spontaneous')
        print('Usage: run_baseline.py list_path kfold audio_type')
        print('Example: python run_baseline.py data/lst 5 phrase_both')
    else:
        list_path = args[0]
        kfold = int(args[1])
        audio_type = args[2]
        main(list_path, kfold, audio_type)
