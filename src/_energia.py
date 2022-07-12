
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
from playsound import playsound
from python_speech_features import mfcc, logfbank, delta
import subprocess, shlex


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
    snd = snd/(2.**15)   
    
    fig = plt.figure(figsize=(8, 8))    
    plt.plot(snd, label="Señal Original")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")    
    plt.show()   
    plt.close()
    
    # calcaulo de la energia
    tramas=windowing(snd, muestreo)
    eng_sum =np.sum(tramas**2, axis=1)    
    

    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum, label="Representacion de la energía")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")   
    plt.show()
    plt.close()
        
    umbral=0.03
    relacion= eng_sum > umbral
    
    fig = plt.figure(figsize=(8, 8))    
    plt.plot(relacion, label="Señal enventanada")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")   
    plt.show()
    plt.close()
    
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum, label="Energía")
    plt.plot(relacion, label="Ventana")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")
    leg = plt.legend(loc="upper right")   
    plt.show()
    plt.close()
    
    
    delta = np.diff(relacion)    
    delta= np.argwhere(delta==True)
    if len(delta)%2!=0:
          delta.append(len(eng_sum)-1)
    
    #Macheo de la señal
    salida=[]    
    for i in  range(0,len(delta)-1,2):
        init= int((delta[i]*Ms)*muestreo)
        end= int((delta[i+1]*Ms)*muestreo)
        salida.append(snd[init : end])
    
    salida = np.concatenate(salida)
    
    fig = plt.figure(figsize=(8, 8))
        
    plt.plot(salida, label="Señal de Salida")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")    
    plt.show()
    plt.close()
    
    waves.write("senal_salida.wav", muestreo, salida)
    return salida ## Yx


    
# START OF THE SCRIPT
if __name__ == "__main__":
    input="/home/frank/sites/unizar/praat/data/audio/AVFAD/test/lolo.wav"
    output= "/home/frank/sites/unizar/praat/data/audio/AVFAD/test/t3.wav"
    command_line = 'ffmpeg -i '+input+' '+output
    args = shlex.split(command_line)
    subprocess.call(args)        
    energia(output)    
   

 