'''
Prosody parameters
'''

from statistics import median
import parselmouth
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import Eng as eng
from parselmouth.praat import call
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
import scipy.io.wavfile as waves
from scipy import signal
from scipy.signal import argrelextrema

# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID)
    
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
        localabsoluteJitter = (call(pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))
        rapJitter = np.log(round(100*call(pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3), 3))
        ppq5Jitter = np.log(round(100*call(pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3), 3))
        ddpJitter = np.log(round(100*call(pulses, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3), 3))
        localShimmer =  np.log(round(100*call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        localdbShimmer = np.log(round(call([sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        apq3Shimmer = np.log(round(100*call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        aqpq5Shimmer = np.log(round(100*call([sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        apq11Shimmer =  np.log(round(100*call([sound, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
        ddaShimmer = np.log(round(100*call([sound, pulses], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 3))
       
       
        columns=["stdev"," hnr"," localJitter"," localabsoluteJitter"," rapJitter"," ppq5Jitter"," ddpJitter"," localShimmer"," localdbShimmer"," apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer", "ddaShimmer"]
        row=[stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
        
    

        return json.dumps(columns), json.dumps((row)) 
    

if __name__ == '__main__':          
    """ args = sys.argv[1:]        
    n =int(args[0]) 
    sound = args[1]
 """
    #sound = "example.wav"
    sound= "data/audio/AVFAD/test/frank.vigil-75c6de05-0cc9-47cc-9b69-d4df79931f0e.m4a.wav"
    data2 = measure2Pitch(sound, 75, 500, "Hertz") 
    
    sound= "data/audio/AVFAD/AAC/AAC002.wav"
    muestreo, snd = waves.read(sound)

    n0 = 65000
    n1 = 75000
    data = snd[n0:n1]
    fig = plt.figure(figsize=(14, 5))
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    
    plt.plot(data)
    plt.grid(True)
    plt.show()
    fig.savefig("data/img/señal_original_zoom.jpg")  # or you can pass a Figure object to pdf.savefig   
    plt.close()


    """ fig = plt.figure(figsize=(14, 5))
    plt.plot(np.arange(len(data)), data)
    plt.plot(signal.argrelextrema(data, np.greater)[0], data[signal.argrelextrema(data, np.greater)], 'o')
    plt.show() """

