
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import scipy.integrate as integrate

from pydub import AudioSegment
from pydub.silence import split_on_silence
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from playsound import playsound
 

#sound= "data/audio/AVFAD/AAC/AAC002.wav"
sound= "data/audio/AVFAD/test/frank.vigil-75c6de05-0cc9-47cc-9b69-d4df79931f0e.m4a.wav"

def eng_origen(sound):
    # INGRESO
    # archivo = input('archivo de audio: ')
    archivo = sound

    # PROCEDIMIENTO
    muestreo, snd = waves.read(archivo)    
    snd = snd/(2.**15)    
    muestra = len(snd)    
    dt = muestra/muestreo
    t = np.arange(0,muestra*dt,dt)    
    s1 = snd[0]       
    # SALIDA - Observación intermedia
    plt.plot(t,snd)
    plt.xlabel('t segundos')
    plt.ylabel('sonido(t)')
    plt.show()
        
    playsound(sound) 
 
def split(filepath):
    sound = AudioSegment.from_wav(filepath)
    dBFS = sound.dBFS
    chunks = split_on_silence(sound, min_silence_len = 500, silence_thresh = dBFS-16,keep_silence = 250 )   
    target_length = 1 * 1000 
    output_chunks = [chunks[0]]
    for i, chunk in enumerate(chunks):
        chunk_name = "bulues{0}.wav".format(i)
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # if the last output chunk is longer than the target length,
            # we can start a new one
            output_chunks.append(chunk)
            chunk.export(chunk_name, format="wav")
   

def windowing(x, fs=16000, Ns=0.025, Ms=0.010):# dividir por tramas las señal
    N = int(Ns * fs)
    M = int(Ms * fs)
    T = len(x)
    m = np.arange(0, T - N, M).reshape(1, -1)
    L = len(m)
    ind = np.arange(N).reshape(-1, 1).dot(np.ones((1, L))) + np.ones((N, 1)).dot(m)
    return x[ind.astype(int).T].astype(np.float32)

def quitarbajas(sound): 
    Ns=0.025
    Ms=0.010       
    muestreo, snd = waves.read(sound)   
    # normalizar la señal del wav
    snd = snd/(2.**15)   
    
    tramas=windowing(snd, muestreo)
    eng_sum =np.sum(tramas**2, axis=1)
    
    
    umbral=0.05
    relacion= eng_sum > umbral
    
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
    
    plt.plot(snd)    
    plt.plot(salida)    
    plt.show()
    
    waves.write("example.wav", muestreo, salida)


#eng_origen(sound)
quitarbajas(sound)