import pandas as pd
from Praat import Praat as praat
from svm_training import main 


def feature_smile():
    main.feature_smile()

### Esta función es para crear los kfold en base a un metadata.xlsx de la base de datos ###
def crea_kfold():
    main.crea_list_kfold()




if __name__ == '__main__':
    feature_smile() #ok
    