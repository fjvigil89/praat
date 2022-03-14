'''
Prosody parameters
'''

from statistics import median
from unittest import case
import parselmouth, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import math
from parselmouth.praat import call

class Praat:    
    path_destinity="data/txt/"
    sound= "data/audio/AVFAD/NORM/1020-a_h.wav"
    #sound= "data/audio/AVFAD/AAC/AAC002.wav"
    
    # This is the function to measure voice pitch
    def measurePitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID) # read the sound

        #pitch = sound.to_pitch()
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)
        
        pulses = call([sound, pitch], "To PointProcess (cc)")
        duration = call(sound, "Get total duration") # duration
        voice_report_str = call([sound, pitch, pulses], "Voice report", 0, 0, 75, 500, 1.3, 1.6, 0.03, 0.45)       
        
        return voice_report_str
    
  
    def measure2Pitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID) # read the sound
        duration = call(sound, "Get total duration") # duration
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)
        pulses = call([sound, pitch], "To PointProcess (cc)")
        
        # "Voice report"       
        meanF0 = np.log(round(call(pitch, "Get mean", 0, 0, unit), 3)) # get mean pitch
        stdevF0 = np.log(round(call(pitch, "Get standard deviation", 0 ,0, unit), 3))# get standard deviation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = np.log(call(harmonicity, "Get mean", 0, 0))
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        
        localJitter = np.log(round(100*call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 3))
        localabsoluteJitter =(call(pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))
        rapJitter = np.log(round(100*call(pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3), 3))
        ppq5Jitter = np.log(round(100*call(pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3), 3))
        ddpJitter = np.log(round(100*call(pulses, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3), 3))
        localShimmer = np.log(round(100*call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        localdbShimmer = np.log(round(call([sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        apq3Shimmer = np.log(round(100*call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        aqpq5Shimmer = np.log(round(100*call([sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        apq11Shimmer = np.log (round(100*call([sound, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        ddaShimmer = np.log(round(100*call([sound, pulses], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
       
       
        columns=["stdev"," hnr"," localJitter"," localabsoluteJitter"," rapJitter"," ppq5Jitter"," ddpJitter"," localShimmer"," localdbShimmer"," apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer", "ddaShimmer"]
        row=[stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
        #return json.dumps(columns), json.dumps((row)) 
        
        return stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
    
    
    """ data = measurePitch(sound, 75, 500, "Hertz")
    file=open(path_destinity+sound.split("/")[4].split(".")[0]+".txt","w")
    file.write(data)
    file.close() """
    # print(data)

    # duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer = measure2Pitch(sound, 75, 500, "Hertz")        
    # lista = [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
    
    # df= pd.DataFrame(lista, index=['duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer'])
    # df.to_excel(path_destinity+sound.split("/")[4].split(".")[0]+"---2.xlsx")
    # # print(df)

    #Sacar la media de todos los valores de una voz NORM
    base = "data/audio/AVFAD/NORM/"
    
    stdev=[]
    hnr0=[]
    localJitter0=[]
    localabsoluteJitter0=[]
    rapJitter0=[]
    ppq5Jitter0=[]
    ddpJitter0=[]
    localShimmer0=[]
    localdbShimmer0=[]
    apq3Shimmer0=[]
    aqpq5Shimmer0=[]
    apq11Shimmer0=[]
    ddaShimmer0=[]

    
    for item in glob.glob(base+'*.wav'):  
        stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer = measure2Pitch(item, 75, 500, "Hertz") 
        if math.isnan(stdevF0):
            print(item)
        if math.isnan(hnr):
            print(item)
        if math.isnan(localJitter):
            print(item)
        if math.isnan(localabsoluteJitter):
            print(item)
        if math.isnan(rapJitter):
            print(item)
        if math.isnan(ppq5Jitter):
            print(item)
        if math.isnan(ddpJitter):
            print(item)
        if math.isnan(localShimmer):
            print(item)
        if math.isnan(localdbShimmer):
            print(item)
        if math.isnan(apq3Shimmer):
            print(item)
        if math.isnan(aqpq5Shimmer):
            print(item)
        if math.isnan(apq11Shimmer):
            print(item)
        if math.isnan(ddaShimmer):
            print(item)
        stdev.append(stdevF0)        
        hnr0.append(hnr)
        localJitter0.append(localJitter)
        localabsoluteJitter0.append(localabsoluteJitter)
        rapJitter0.append(rapJitter)
        ppq5Jitter0.append(ppq5Jitter)
        ddpJitter0.append(ddpJitter)
        localShimmer0.append(localShimmer)
        localdbShimmer0.append(localdbShimmer)
        apq3Shimmer0.append(apq3Shimmer)
        aqpq5Shimmer0.append(aqpq5Shimmer)
        apq11Shimmer0.append(apq11Shimmer)
        ddaShimmer0.append(ddaShimmer)
    

    rest=[
        np.median(stdev),
        np.median(hnr0),
        np.median(localJitter0),
        np.median(localabsoluteJitter0),
        np.median(rapJitter0),
        np.median(ppq5Jitter0),
        np.median(ddpJitter0),
        np.median(localShimmer0),
        np.median(localdbShimmer0),
        np.median(apq3Shimmer0),
        np.median(aqpq5Shimmer0),
        np.median(apq11Shimmer0),
        np.median(ddaShimmer0),

    ]
    std_mas=[
        np.median(stdev)+np.std(stdev),
        np.median(hnr0)+ np.std(hnr0),
        np.median(localJitter0)+ np.std(localJitter0),
        np.median(localabsoluteJitter0)+ np.std(localabsoluteJitter0),
        np.median(rapJitter0)+ np.std(rapJitter0),
        np.median(ppq5Jitter0)+ np.std(ppq5Jitter0),
        np.median(ddpJitter0)+ np.std(ddpJitter0),
        np.median(localShimmer0)+ np.std(localShimmer0),
        np.median(localdbShimmer0)+ np.std(localdbShimmer0),
        np.median(apq3Shimmer0)+ np.std(apq3Shimmer0),
        np.median(aqpq5Shimmer0)+ np.std(aqpq5Shimmer0),
        np.median(apq11Shimmer0)+ np.std(apq11Shimmer0),
        np.median(ddaShimmer0)+ np.std(ddaShimmer0),

    ]
    std_menos=[
        np.median(stdev)-np.std(stdev),
        np.median(hnr0)- np.std(hnr0),
        np.median(localJitter0)- np.std(localJitter0),
        np.median(localabsoluteJitter0)- np.std(localabsoluteJitter0),
        np.median(rapJitter0)- np.std(rapJitter0),
        np.median(ppq5Jitter0)- np.std(ppq5Jitter0),
        np.median(ddpJitter0)- np.std(ddpJitter0),
        np.median(localShimmer0)- np.std(localShimmer0),
        np.median(localdbShimmer0)- np.std(localdbShimmer0),
        np.median(apq3Shimmer0)- np.std(apq3Shimmer0),
        np.median(aqpq5Shimmer0)- np.std(aqpq5Shimmer0),
        np.median(apq11Shimmer0)- np.std(apq11Shimmer0),
        np.median(ddaShimmer0)- np.std(ddaShimmer0),

    ]
    print(rest)
    print(std_mas)
    print(std_menos)
     