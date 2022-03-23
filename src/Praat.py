'''
Prosody parameters
'''

from statistics import median
import parselmouth
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from parselmouth.praat import call
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)


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
    
def praat(n, data):
    valor=[]
    valor.append(data[62:82])    
    valor.append(data[89:110])
    valor.append(data[115:142])
    valor.append(data[148:170])
    valor.append(data[174:199])        
    valor.append(data[213:235])       
    valor.append(data[238:261])
    valor.append(data[264:289])
    valor.append(data[300:342])    
    valor.append(data[360:406])
    valor.append(data[423:451])
    valor.append(data[452:485])    
    valor.append(data[538:559])    
    valor.append(data[564:600])    
    valor.append(data[611:631])
    valor.append(data[635:656])    
    valor.append(data[660:680])    
    valor.append(data[693:716])    
    valor.append(data[721:747])    
    valor.append(data[754:775])    
    valor.append(data[780:801])
    valor.append(data[806:828])    
    valor.append(data[833:853])    
    valor.append(data[896:926])
    valor.append(data[930:969])
    valor.append(data[973:1010])
    
    columns=[]
    row=[]
    
    for item in valor:        
        columns.append(item.split(':')[0])
        row.append(float(item.split(':')[1]))  

    
    return json.dumps(columns), json.dumps(row)  

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")


if __name__ == '__main__':          
    """ args = sys.argv[1:]        
    n =int(args[0]) 
    sound = args[1]
 """
    #sound = "example.wav"
    sound= "data/audio/AVFAD/test/frank.vigil-75c6de05-0cc9-47cc-9b69-d4df79931f0e.m4a.wav"
    data2 = measure2Pitch(sound, 75, 500, "Hertz") 
    
    #data = praat(n, data)
    #print(data)

    # print(data2)
    
    sound= "data/audio/AVFAD/AAC/AAC002.wav"
    # snd = parselmouth.Sound(sound)
    # intensity = snd.to_intensity()
    # spectrogram = snd.to_spectrogram()
    # plt.figure()
    # draw_spectrogram(spectrogram)
    # plt.twinx()
    # draw_intensity(intensity)
    # plt.xlim([snd.xmin, snd.xmax])
    # plt.show() # or plt.savefig("spectrogram.pdf")
    
