'''
Prosody parameters
'''

from statistics import median
import parselmouth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parselmouth.praat import call

class Praat:    
    path_destinity="data/txt/"
    #sound= "data/audio/AVFAD/AAO/AAO001.wav"
    sound= "data/audio/AVFAD/AAC/AAC002.wav"
    
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
        # pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.015, 0.35, 0.14, f0max)
        # pulses = call([sound, pitch], "To PointProcess (cc)")
        # voice_report_str = call([sound, pitch, pulses], "Voice report", 5.144440, 6.418696, 75, 500, 1.3, 1.6, 0.03, 0.45)       
        meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = 100*call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = 100*call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = 100*call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = 100*call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer =  100*call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = 100*call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = 100*call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer =  100*call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = 100*call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
       
        
        return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
        
    data = measurePitch(sound, 75, 500, "Hertz")
    file=open(path_destinity+sound.split("/")[4].split(".")[0]+".txt","w")
    file.write(data)
    file.close()
    # print(data)

    # duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer = measure2Pitch(sound, 75, 500, "Hertz")        
    # lista = [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
    
    # df= pd.DataFrame(lista, index=['duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer'])
    # df.to_excel(path_destinity+sound.split("/")[4].split(".")[0]+"---2.xlsx")
    # # print(df)
    
    
    
    