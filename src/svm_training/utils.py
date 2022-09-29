import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

def zscore(x,u=False,s=False): 
    sx,feat, = x.shape
    xnorm = np.zeros((sx,feat))    
    try:
        test = u.shape #Just for invalidate the try if u==False
        xnorm = (x - u)/s
        return xnorm
    except: 
        u = np.mean(x,axis=0)                
        s = np.zeros(feat)
        for i in range(0,feat):
            s[i] = np.std(x[:,i]) + 1e-20              
        xnorm = (x - u)/s
        return xnorm, u, s


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
def z_standard(df):
    # create a scaler object
    std_scaler = StandardScaler()
    std_scaler
    # fit and transform the data
    df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
    return df_std


def split_data(path, labels):
    path_csv = path
    df = pd.read_csv(path_csv)
    df = df.assign(label=labels)
    
    data = df.drop(columns=['file', 'start', 'end'])      
    #"all PATH(1) or NORM(0)")
    X = data.drop(columns=['label'], axis=1)
    y = data['label']
    X = X.astype(float).fillna(0.0)
    # print(data['label'].value_counts())

    #"========== normalization z_score =============")      
    X = z_standard(X)
    return X, y

def compute_score(model, test_lbltrue, test_lblpredict, test_features, roc=False):
    # output: score (has 13 metrics values)  
    # score = 0.acc, 1.acc0, 2.acc1, 3.uar, 4.f1score, 5.recall, 6.precision, 7.auc, 8.eer, 9.tp, 10.tn, 11.fp, 12.fp
    score = []
    score.append(accuracy_score(test_lbltrue, test_lblpredict))
    score.append(accuracy_score(test_lbltrue[test_lbltrue==0], test_lblpredict[test_lbltrue==0]))
    score.append(accuracy_score(test_lbltrue[test_lbltrue==1], test_lblpredict[test_lbltrue==1]))
    score.append(balanced_accuracy_score(test_lbltrue, test_lblpredict))
    score.append(f1_score(test_lbltrue,test_lblpredict))
    score.append(recall_score(test_lbltrue,test_lblpredict))
    score.append(precision_score(test_lbltrue,test_lblpredict))
    scores=model.predict_proba(test_features)[:,1]
    fpr, tpr, thresholds = roc_curve(test_lbltrue, scores)
    score.append(roc_auc_score(test_lbltrue, scores))
    eer_value=brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    score.append(eer_value)
    if roc:
        plt.figure()
        plt.plot(fpr,tpr)
        plt.title('EER='+str(eer_value))
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.grid(True)
        plt.savefig('roc_curve.png')
    tn, fp, fn, tp = confusion_matrix(test_lbltrue, test_lblpredict).ravel()
    N = tn + fp + fn + tp
    score.append(tp/N * 100)
    score.append(tn/N * 100)
    score.append(fp/N * 100)
    score.append(fn/N * 100) 
    return np.array(score) 
    
    
def compute_score_multiclass(test_lbltrue, test_lblpredict, data, resumen):
    
    # labelsnames = data['labels'].keys()
    # lbl = {}
    # for i in labelsnames:
    #     lbl[data['labels'][i]] = i
    lbl = data
    aux1 = set(unique_labels(test_lbltrue, test_lblpredict))
    aux2 = set(np.unique(test_lbltrue))
    while aux1 != aux2:
        for i in aux1:
            if i not in aux2:
                temp = np.where(test_lblpredict == i)
                test_lbltrue[temp] = test_lblpredict[temp]

        aux2 = set(np.unique(test_lbltrue))
        aux1 = set(unique_labels(test_lbltrue, test_lblpredict))


    labels = np.unique(test_lbltrue)
    #target_names = []
    target_names = data
    # for i in labels:
    #     target_names.append(lbl[i])

    out = classification_report(test_lbltrue, test_lblpredict, labels=labels, target_names=target_names, output_dict = resumen, zero_division=0)

    return out

