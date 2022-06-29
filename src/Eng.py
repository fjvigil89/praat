
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
from playsound import playsound
from python_speech_features import mfcc, logfbank, delta


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
 
def load_audios(input_path): #devuelve frecuencia, audio, y tiempo 
    audio = waves.read(input_path)
    samples = np.array(audio[1], dtype=np.float32)
    Fs = audio[0]

    muestra = len(samples)    
    dt = muestra/Fs
    t = np.arange(0,muestra*dt,dt) 
    return  Fs, samples, t


def normalice_wav(x):
    x_scaled = x/(2.**15)
    return x_scaled


def save_audios(output_path, samples, Fs, scaling=True):
    if scaling:
        samples_scaled = normalice_wav(samples)
    else:
        samples_scaled = samples
    waves.write(output_path, Fs, samples_scaled)


def show_signal(samples):
    t = np.ones(len(samples))
    t = np.cumsum(t)
    t = t-1
    plt.plot(t, samples)
    plt.grid()
    plt.show()


def windowing(x, fs=16000, Ns=0.025, Ms=0.010):# tipo rectángulo

    N = int(Ns * fs)
    M = int(Ms * fs)
    T = len(x)
    m = np.arange(0, T - N, M).reshape(1, -1)
    L = len(m)
    ind = np.arange(N).reshape(-1, 1).dot(np.ones((1, L))) + np.ones((N, 1)).dot(m)

    return x[ind.astype(int).T].astype(np.float32)


def window(x, N): #tipo Hanning
    t = np.arange(0, N)
    xwi = x * (0.5)*(1-np.cos(  np.pi*t/((N-1)/2)  ))
    return xwi

def wHamming(x, N): #tipo Hamming
    t = np.arange(0, N)
    xwi = x * (0.46164)*(1-np.cos(  np.pi*t/((N-1)/2)  ))
    return xwi 

def quitarbajas(sound): 
    Ns=0.025
    Ms=0.010       
    muestreo, snd = waves.read(sound)   
    # normalizar la señal del wav
    snd = snd/(2.**15)   
    
    fig = plt.figure(figsize=(8, 8))    
    plt.plot(snd, label="Señal Original")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    fig.savefig("data/img/señal_original.jpg")  # or you can pass a Figure object to pdf.savefig 
    plt.show()   
    plt.close()
    
    # calcaulo de la energia
    tramas=windowing(snd, muestreo)
    eng_sum =np.sum(tramas**2, axis=1)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum, label="Representacion de la energía")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")
    fig.savefig("data/img/señal_original_energia.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.show()
    plt.close()
        
    umbral=0.03
    relacion= eng_sum > umbral
    
    fig = plt.figure(figsize=(8, 8))    
    plt.plot(relacion, label="Señal enventanada")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")
    fig.savefig("data/img/señal_original_ventanas.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.show()
    plt.close()
    
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum, label="Energía")
    plt.plot(relacion, label="Ventana")
    plt.xlabel("Tramas de energia")
    plt.ylabel("Amplitud normalizada")
    leg = plt.legend(loc="upper right")
    fig.savefig("data/img/señal_original_ventanas_Original.jpg")  # or you can pass a Figure object to pdf.savefig
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
    fig.savefig("data/img/señal_salida.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.show()
    plt.close()
    
    waves.write("senal_salida.wav", muestreo, salida)
    
    return salida ## Yx

def ruido(sound):      
    output_path= "noisy.wav"  
    muestreo, snd, t = load_audios(sound)  
    snd = normalice_wav(snd)

    # calcaulo de la energia
    Ms=0.010 
    # calcaulo de la energia
    tramas=windowing(snd, muestreo)
    eng_sum =np.sum(tramas**2, axis=1)
        
    umbral=0.05    
    
    relacion=[False,*(eng_sum <= umbral), False]
    # print("relacion", relacion)
    
    delta = np.diff(relacion)        
    delta= np.argwhere(delta==True)  
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(relacion)
    fig.savefig("data/img/esp_original_ventanas.jpg")  # or you can pass a Figure object to pdf.savefig    
    plt.close()  
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum)
    plt.plot(relacion)
    fig.savefig("data/img/esp_original_ventanas_Original.jpg")  # or you can pass a Figure object to pdf.savefig
    # plt.show()
    plt.close()
  
    
    #Macheo de la señal
    salida=[]    
    for i in  range(0,len(delta)-1,2):        
        init= int((delta[i]*Ms)*muestreo)
        end= int((delta[i+1]*Ms)*muestreo)
        salida.append(snd[init : end])

    
    Vx = np.concatenate(salida)
    
    fig = plt.figure(figsize=(8, 8))
    
    
    plt.plot(Vx, label="Área debajo del Umbral")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    fig.savefig("data/img/Vx_Ruido.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.show()
    plt.close()
    
    save_audios(output_path, Vx, muestreo)
    return Vx, output_path  



    
# START OF THE SCRIPT
if __name__ == "__main__":
    # Audio files paths
    
    sound= "data/audio/AVFAD/AAC/AAC002.wav"
    #sound= "data/audio/AVFAD/test/frank.vigil-75c6de05-0cc9-47cc-9b69-d4df79931f0e.m4a.wav"
    #sound= "data/audio/AVFAD/test/TVD-D-0021_D1_A_1.wav" 
    quitarbajas(sound)    
    #Vx, noisy_path = ruido(sound)
    #output_path = "filtered.wav"

 