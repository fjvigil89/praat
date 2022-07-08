import os, sys, pickle, json, time

from sklearn.svm import SVC
from utils import compute_score, zscore
import pandas as pd
import numpy as np
import opensmile

ker = 'poly'
d = 1
c = 1


def zscore(x, u=False, s=False):
    feat = len(x)
    try:
        # test = u.shape  # Just for invalidate the try if u==False
        xnorm = (x - u)/s
        return xnorm
    except:
        u = np.mean(x,axis=0)
        s = np.zeros(feat)
        for i in range(0,feat):
            s[i] = np.std(x[:,i]) + 1e-20
        xnorm = (x - u)/s
        return xnorm, u, s


def load_svm_model():
    model = pickle.load(open("src/svm_classification/models/SVM_Model.pkl", 'rb'))
    trainmean = pickle.load(open("src/svm_classification/models/Stat_Mean.pkl", 'rb'))
    trainstd = pickle.load(open("src/svm_classification/models/Stat_STD.pkl", 'rb'))

    return model, trainmean, trainstd


class SVM_classification:

    def extract_parameters(audio_path, outpath):
        # opensmile only works with single channel wav files
        # This function work with two input data (path to wav file and path to output parameters (csv format))
        # extract_parameters function return the parameters in numpy array
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        smileparam = smile.process_file(audio_path)
        smileparam.to_csv(outpath)

        feat = pd.read_csv(outpath).to_numpy()[0]
        features = feat[3:]

        return features

    def svm_predict(features, model, mstat, sstat):
        norm_features = zscore(features, mstat, sstat)
        out = model.predict(np.transpose(norm_features.reshape(-1, 1)))

        return out

    if __name__ == '__main__':
        # args = sys.argv[1:]                
        # sound = args[0]

        sound = "data/audio/dataset_TuVoz/TVD-T-0001/TVD-T-0001_8.wav"           
        out_put = "test.csv"

        parameters = extract_parameters(sound, out_put)
        model, mean_stat, std_stat = load_svm_model()
        out_predict = svm_predict(parameters, model, mean_stat, std_stat)
        print(out_predict)
        # score = compute_score(model, [1], out_predict, parameters)
