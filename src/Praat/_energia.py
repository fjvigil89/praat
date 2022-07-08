
from wsgiref.util import request_uri
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from Praat import Praat as praat
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
from playsound import playsound
from python_speech_features import mfcc, logfbank, delta
from datetime import date, datetime
from sklearn.mixture import GaussianMixture

def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a[0], b[0], c[0]])

def windowing(x, fs=16000, Ns=0.025, Ms=0.010):# tipo rectángulo

    N = int(Ns * fs)
    M = int(Ms * fs)
    T = len(x)
    m = np.arange(0, T - N, M).reshape(1, -1)
    L = len(m)
    ind = np.arange(N).reshape(-1, 1).dot(np.ones((1, L))) + np.ones((N, 1)).dot(m)

    return x[ind.astype(int).T].astype(np.float32)

def energia(sound): 
    Ns=0.025
    Ms=0.010       
    muestreo, snd = waves.read(sound)      
        
    # normalizar la señal del wav    
    # snd = snd - np.mean(snd)
    # snd = snd / float(np.max(np.abs(snd)))
    snd = snd/(2.**15)   
    
    # calcaulo de la energia
    tramas=windowing(snd, muestreo, Ns, Ms)
    eng_sum =np.sum(tramas**2, axis=1)    
    #eng_sum = 10*np.log10(eng_sum)  # Convertir a dB
    
    plt.plot(eng_sum, label="Representacion de la energía")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")    
    plt.show()
    plt.close()

    # q3_q, q1_q = np.quantile(np.sort(eng_sum), [0.95, 0.25])
    # umbral = q3_q - q1_q        
    # #umbral=3
    # relacion= eng_sum > umbral
    gm = GaussianMixture(n_components=2, random_state=0).fit(eng_sum.reshape(-1, 1))
    medias = gm.means_
    varianza = gm.covariances_
    std_var = np.sqrt(varianza)

    curve_cuts = solve(medias[0][0], medias[1][0], std_var[0][0], std_var[1][0])
    umbral = 0
    for i in curve_cuts:
        if np.min(medias) < i < np.max(medias):
            umbral = i

    relacion = eng_sum > umbral
          
    plt.plot(relacion, label="Señal enventanada")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")    
    plt.show()
    plt.close()
    
    
   
    plt.plot(eng_sum, label="Energía")
    plt.plot(relacion, label="Ventana")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")
    plt.show()
    plt.close()    
    
    
    delta = np.diff(relacion)
    delta= np.argwhere(delta==True)
    if len(delta)%2!=0:
          np.append(delta,len(eng_sum)-1)
    
    
    #Macheo de la señal
    salida=[]        
    for i in  range(0,len(delta)-1,2):
        init= int((delta[i]*Ms)*muestreo)
        end= int((delta[i+1]*Ms)*muestreo)
        salida.append(snd[init : end])
    
    salida = np.concatenate(salida)    
        
    outpath="output_"+str(date.today())+".wav"    
    waves.write(outpath, muestreo, salida)
    #data= praat.measurePitch(outpath,75, 500, "Hertz")   
    #print(data)
    return

    
# START OF THE SCRIPT
if __name__ == "__main__":    
    sound= "data/audio/tmp/TVD-T-00010_4.wav"
    energia(sound)  
    
    
   

 