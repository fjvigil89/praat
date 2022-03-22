
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import scipy.integrate as integrate
import scipy.signal as signal 

from pydub import AudioSegment
from pydub.silence import split_on_silence
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
from playsound import playsound
 

sound= "data/audio/AVFAD/AAC/AAC002.wav"
#sound= "data/audio/AVFAD/test/frank.vigil-75c6de05-0cc9-47cc-9b69-d4df79931f0e.m4a.wav"

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
    
    waves.write("senal_salida.wav", muestreo, salida)


def ruido(sound):
    ## Compute Fourier Transform
    # PROCEDIMIENTO
    Ms=0.010  
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

    print((threshold))
    """ plt.plot(fhat_clean)
    plt.show() """
    signal_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(signal_filtered)
    fig.savefig("data/img/señal_si_ruido.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
   
    waves.write("example_sinRuido.wav", muestreo, signal_filtered)
    
def test(sound):   
    sampling_rate, snd = waves.read(sound)      
    snd = snd/(2.**15)        
    muestra = len(snd)    
    dt = muestra/sampling_rate    
    t = np.arange(0,muestra*dt,dt)  
    n = len(t)
    f2=50
    fft = np.fft.fft(np.abs(snd),n)    
    fft_size = len(fft)# Longitud de muestreo de procesamiento #FFT
    
    
    
    x = snd # Se superponen dos ondas sinusoidales, 156.25HZ y 234.375HZ
    # El requisito de FFT de N puntos para un análisis de espectro preciso es que N puntos de muestreo contengan un número entero de objetos de muestreo. Por lo tanto, la FFT de N puntos puede calcular perfectamente el espectro. El requisito para el objeto de muestreo es n * Fs / N (n * frecuencia de muestreo / longitud de FFT),
    # Por lo tanto, para 8 KHZ y 512 puntos, el requisito mínimo para el período de un objeto de muestreo perfecto es 8000/512 = 15,625 HZ, por lo que el n de 156,25 es 10 y el n de 234,375 es 15.
    xs = x[:fft_size]# Muestreo de puntos de fft_size de los datos de forma de onda para el cálculo
    xf = np.fft.rfft(xs)/fft_size   # Use np.fft.rfft () para el cálculo de FFT, rfft () es para una transformación más 
                                    # conveniente de señales de números reales, a partir de la fórmula podemos 
                                    # ver / fft_size para mostrar correctamente la energía de la forma de onda
                                    
    # El valor de retorno de la función rfft es N / 2 + 1 números complejos, que representan puntos desde 0 (Hz) 
    # a sample_rate / 2 (Hz).
    #De modo que la frecuencia verdadera correspondiente a cada subíndice en el valor de retorno se puede calcular 
    #mediante el siguiente np.linspace:
    freqs = np.linspace(0, sampling_rate/2, int(fft_size/2+1))
    
    # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    #Return números espaciados uniformemente dentro del intervalo especificado
    xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    # Por último, calculamos la amplitud de cada componente de frecuencia y la convertimos a un valor en db hasta 20 * np.log10 (). Para evitar que el componente de amplitud cero haga que log10 no se calcule, llamamos a np.clip para realizar el procesamiento de límite superior e inferior en la amplitud de xf

    psd = fft * np.conj(fft)/n
    threshold = -200
    psd_idxs = psd > threshold #array of 0 and 1
    print("psd_idxs", psd_idxs)
    
    #Drawing display results
    plt.figure(figsize=(8,4))
    plt.subplot(211)
    plt.plot(t[:fft_size], xs)
    plt.xlabel(u"Time(S)")
    plt.title(u"156.25Hz and 234.375Hz WaveForm And Freq")   
    
    plt.subplot(212)
    plt.plot(freqs, xfp)
    plt.xlabel(u"Freq(Hz)")
    plt.subplots_adjust(hspace=0.4)
   
    
    plt.show()

#quitarbajas(sound)
#test("senal_salida.wav")

# Transformada de Fourier
def fft1(xx):
#   t=np.arange(0, s)
    t=np.linspace(0, 1.0, len(xx))
    f = np.arange(len(xx)/2+1, dtype=complex)
    for index in range(len(f)):
        f[index]=complex(np.sum(np.cos(2*np.pi*index*t)*xx), -np.sum(np.sin(2*np.pi*index*t)*xx))
    return f

