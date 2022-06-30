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

class Praat():    
    def measurePitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID)
        
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)
        
        pulses = call([sound, pitch], "To PointProcess (cc)")
        start_time = call(pulses, "Get time from index", 1)
        end_time = call(pulses, "Get time from index", call(pulses, "Get number of points"))
        duration = call(sound, "Get total duration") # duration
        voice_report_str = call([sound, pitch, pulses], "Voice report", 0.768239, 1.258846, 75, 500, 1.3, 1.6, 0.03, 0.45)       
        
        #print(start_time+0.2)
        return voice_report_str

    def measure2Pitch(voiceID, f0min, f0max, unit):
            sound = parselmouth.Sound(voiceID) # read the sound
            duration = call(sound, "Get total duration") # duration
            pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)
            pulses = call([sound, pitch], "To PointProcess (cc)")
            
            print(pulses)
            start_time = call(pulses, "Get time from index", 1)
            end_time = call(pulses, "Get time from index", call(pulses, "Get number of points"))
            
            # "Voice report"       
            meanF0 = (round(call(pitch, "Get mean", 0, 0, unit), 3)) # get mean pitch
            stdevF0 = (round(call(pitch, "Get standard deviation", 0 ,0, unit), 3))# get standard deviation
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
            hnr = (call(harmonicity, "Get mean", 0, 0))
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
            
            localJitter = (round(100*call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 3))
            localabsoluteJitter = (call(pulses, "Get jitter (local, absolute)", 0,0, 0.0001, 0.02, 1.3))
            rapJitter = (round(100*call(pulses, "Get jitter (rap)", 0,0, 0.0001, 0.02, 1.3), 3))
            ppq5Jitter = (round(100*call(pulses, "Get jitter (ppq5)", 0,0, 0.0001, 0.02, 1.3), 3))
            ddpJitter = (round(100*call(pulses, "Get jitter (ddp)", 0,0, 0.0001, 0.02, 1.3), 3))
            localShimmer =  (round(100*call([sound, pulses], "Get shimmer (local)", 0,0, 0.0001, 0.02, 1.3, 1.6), 3))
            localdbShimmer = (round(call([sound, pulses], "Get shimmer (local_dB)", 0,0, 0.0001, 0.02, 1.3, 1.6), 3))
            apq3Shimmer = (round(100*call([sound, pulses], "Get shimmer (apq3)", 0,0, 0.0001, 0.02, 1.3, 1.6), 3))
            aqpq5Shimmer = (round(100*call([sound, pulses], "Get shimmer (apq5)", 0,0, 0.0001, 0.02, 1.3, 1.6), 3))
            apq11Shimmer =  (round(100*call([sound, pulses], "Get shimmer (apq11)", 0,0, 0.0001, 0.02, 1.3, 1.6), 3))
            ddaShimmer = (round(100*call([sound, pulses], "Get shimmer (dda)", 0,0, 0.0001, 0.02, 1.3, 1.6), 3))
        
        
            columns=["meanF0","stdev"," hnr"," localJitter"," localabsoluteJitter"," rapJitter"," ppq5Jitter"," ddpJitter"," localShimmer"," localdbShimmer"," apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer", "ddaShimmer"]
            row=[meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]
            
        

            return json.dumps(columns), json.dumps((row)) 
        
    def amplitud_periodo(sound):
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
        
    if __name__ == '__main__':          
        """ args = sys.argv[1:]        
        n =int(args[0]) 
        sound = args[1]
    """
        sound= "data/audio/tmp/TVD-T-00010_4.wav"        
        #sound="senal_salida.wav"
        data2 = measure2Pitch(sound, 75, 500, "Hertz") 
        print(data2)
        
        


