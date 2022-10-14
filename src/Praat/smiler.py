# Tratamiento de datos
# ==============================================================================

from sklearn.preprocessing import StandardScaler
import warnings
import sys 
import json
import opensmile
import pandas as pd
import numpy as np
import time

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score, precision_score, f1_score, recall_score, average_precision_score
# ==============================================================================
warnings.filterwarnings('ignore')

#sound= "data/audio/AVFAD/AAO/AAO001.wav"
sound= "data/audio/AVFAD/AAC/AAC001.wav"
path_destinity="data/txt/"

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
    loglevel=2,
    logfile='smile.log',
)

feature = smile.process_files([sound])
feature.to_excel(path_destinity+sound.split("/")[4].split(".")[0]+".xlsx") 