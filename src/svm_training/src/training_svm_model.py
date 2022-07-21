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


if __name__ == '__main__':
        sound = "/home/invitados/freyes/Flavio/Esperanto/BD/THALENTO_PROCESS/TVD-Disorder/TVD-D-0001/pre/procesadas/TVD-D-0001_D8_LECTURA.wav"
        # sound = "/home/invitados/freyes/Flavio/Esperanto/Experiment_folder/exp_svm_clasification/baseline/data/audio/thalento_vowels/TVD-P-0001-AIU.wav"
        out_put = "test.csv"

        parameters = extract_parameters(sound, out_put)
        # print(out_predict)
        # score = compute_score(model, [1], out_predict, parameters)
