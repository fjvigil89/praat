# Tratamiento de datos
# ==============================================================================
import warnings
import json
import opensmile
import pandas as pd

# ==============================================================================
warnings.filterwarnings('ignore')

destinity = "data/csv/"
data_path = "data/audio/Saarbruecken"

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
    loglevel=2,
    logfile='smile.log',
)

# list=['phrase','a_n','aiu_n','lhl','aiu']
list=['phrase','a_n','aiu_nlh']
nfold = 5
while nfold > 0:      
      for x in list: 
          print("Procesing data/lst/train_"+x+"_both_meta_data_fold"+str(nfold)+".json")                                        
          path_json = "data/lst/train_"+x+"_both_meta_data_fold"+str(nfold)+".json"
          with open(path_json,'r') as f:
            data = json.loads(f.read())
            df = pd.json_normalize(data, record_path =['meta_data'])
            
          # read wav files and extract emobase features on that file        
          feature = smile.process_files(df['path']) 
          feature = feature.assign(type=x)
          feature = feature.assign(label="")
          feature.to_csv(destinity+"train_"+x+"_both_meta_data_fold"+str(nfold)+".csv")                               
        
        #=====Test=====
          path_json = "data/lst/test_"+x+"_both_meta_data_fold"+str(nfold)+".json"
          with open(path_json,'r') as f:
            data = json.loads(f.read())
            df = pd.json_normalize(data, record_path =['meta_data'])
          
          # read wav files and extract emobase features on that file
          feature = smile.process_files(df['path'])  
          feature = feature.assign(type=x)
          feature = feature.assign(label="")
          feature.to_csv(destinity+"test_"+x+"_both_meta_data_fold"+str(nfold)+".csv")    
          
                              
      nfold -= 1 

