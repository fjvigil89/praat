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
import librosa
import librosa.display
import python_speech_features
import warnings

from parselmouth.praat import call
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
from spectrum import Spectrogram, dolphin_filename, readwav
from scipy.io.wavfile import read


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
    
    sound= "data/audio/AVFAD/AAC/AAC002.wav"    
    scale, sr= librosa.load(sound)    
    
    # Mel Filter
    filter_banck = librosa.filters.mel(n_fft=2048, sr=sr,n_mels=10)    
    print(filter_banck.shape)
    fig= plt.figure(figsize=(15, 8))
    librosa.display.specshow(filter_banck, sr=sr, x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.xlabel("Frecuencia Líneal (Hz)")
    plt.ylabel("Frecuencia Mel (mel)")
    
    fig.savefig("data/img/banco_filtro_mel.jpg")  # or you can pass a Figure object to pdf.savefig 
    # plt.show()   
    plt.close()
    
    # Extracting Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10 )
    print(mel_spectrogram.shape)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    print(log_mel_spectrogram.shape)
    
    fig= plt.figure(figsize=(15, 8))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.f")
    plt.xlabel("Tiempo")
    plt.ylabel("Frecuencia (Hz)")
    
    fig.savefig("data/img/log_mel_spectrogram.jpg")  # or you can pass a Figure object to pdf.savefig 
    # plt.show()   
    plt.close()
    
   
    
    