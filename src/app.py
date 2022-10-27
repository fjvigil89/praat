import pandas as pd
from Praat import Praat as praat
from svm_training import main, run_baseline 
import datetime

def feature_smile(): #ok
    main.feature_smile()

### Esta función es para crear los kfold en base a un metadata.xlsx de la base de datos ###
def crea_kfold():
    main.crea_list_kfold()

def make_model_svc():
    main.train_svm()

def make_model_flavio():
    main.train_svm_flavio()
    
def tiempo_total():
    main.tiempo_total()

def tiempo_total_pathology():
    main.tiempo_total_pathology()

def tiempo_total_audio():
    main.tiempo_total_audio()


if __name__ == '__main__':
    #crea_kfold() #revisar en las 4 DB
    
    #feature_smile() #ok 1 
    make_model_flavio() #ok 2
    
    #tiempo_total() #ok 3
    # tiempo_total_pathology() #ok
    #tiempo_total_audio()
    
    
