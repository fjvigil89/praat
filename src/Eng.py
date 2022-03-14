
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import scipy.integrate as integrate
import scipy.signal as signal 

from pydub import AudioSegment
from pydub.silence import split_on_silence
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from playsound import playsound
 

sound= "data/audio/AVFAD/AAC/AAC002.wav"

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
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(snd)
    fig.savefig("data/img/señal_original.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
    
    # calcaulo de la energia
    tramas=windowing(snd, muestreo)
    eng_sum =np.sum(tramas**2, axis=1)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum)
    fig.savefig("data/img/señal_original_energia.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
        
    umbral=0.05
    relacion= eng_sum > umbral
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(relacion)
    fig.savefig("data/img/señal_original_ventanas.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
    
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(eng_sum)
    plt.plot(relacion)
    fig.savefig("data/img/señal_original_ventanas_Original.jpg")  # or you can pass a Figure object to pdf.savefig
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
    plt.plot(salida)
    fig.savefig("data/img/señal_salida.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
    
    waves.write("example.wav", muestreo, salida)

def ruido(sound):
    ## Compute Fourier Transform
    # PROCEDIMIENTO
    muestreo, snd = waves.read(sound)    
    snd = snd/(2.**15)    
    muestra = len(snd)    
    dt = muestra/muestreo
    t = np.arange(0,muestra*dt,dt)  
    n = len(t)
    
    fhat = np.fft.fft(snd, n) #computes the fft
    psd = fhat * np.conj(fhat)/n
    idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32) #first half index
    psd_real = np.abs(psd[idxs_half]) #amplitude for first half


    ## Filter out noise
    sort_psd = np.sort(psd_real)[::-1]
    # print(len(sort_psd))
    threshold = sort_psd[12]
    psd_idxs = psd > threshold #array of 0 and 1
    psd_clean = psd * psd_idxs #zero out all the unnecessary powers
    fhat_clean = psd_idxs * fhat #used to retrieve the signal

    signal_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(signal_filtered)
    fig.savefig("data/img/señal_si_ruido.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
    
    waves.write("example_sinRuido.wav", muestreo, signal_filtered)
    
def test(sound):        
    muestreo, snd = waves.read(sound)        
    f1 = 25 
    f2 = 50 
    N = 10 
    
    # t = np.linspace(0, 1, 1000) 
    muestra = len(snd)    
    dt = muestra/muestreo
    t = np.arange(0,muestra*dt,dt)    
    sig = snd +  np.sin(2*np.pi*f2*t)
    
    fig,(ax1, ax2) = plt.subplots(2, 1, sharex=True) 
    ax1.plot(t, sig) 
    ax1.set_title('25 Hz and 50 Hz sinusoids') 
    ax1.axis([0, 1, -2, 2]) 
    
    sos = signal.butter(50, 35, 'lp', fs=muestreo, output='sos') 
    
    filtered = signal.sosfiltfilt(sos, sig) 
    
    
    ax2.plot(t, filtered) 
    ax2.set_title('After 35 Hz Low-pass filter') 
    ax2.axis([0, 1, -2, 2]) 
    ax2.set_xlabel('Time [seconds]') 
    plt.tight_layout() 
    plt.show() 
    


#quitarbajas(sound)
ruido("example.wav")