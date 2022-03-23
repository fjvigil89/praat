
from math import gamma
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

def window(x, N): #tipo Arco
    t = np.arange(0, N)
    xwi = x * (0.5)*(1-np.cos(  np.pi*t/((N-1)/2)  ))
    return xwi

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
    relacion= eng_sum <= umbral 

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
    plt.close()
  
    if len(delta)%2!=0:
          delta.append(len(eng_sum)-1)


    gama=[]
    gama.append(eng_sum[0])
    for i in  range(0,len(delta)-1):
        gama.append(delta[i])  
    gama.append(eng_sum[len(eng_sum)-1])
    
    
    #Macheo de la señal
    salida=[]    
    for i in  range(0,len(gama)-1,2):        
        init= int((gama[i]*Ms)*muestreo)
        end= int((gama[i+1]*Ms)*muestreo)
        salida.append(snd[init : end])

    
    Vx = np.concatenate(salida)
    

    save_audios(output_path, Vx, muestreo)
    return Vx, output_path  

def power_spectral_density_estimation_of_the_noisy_signal(Yi, N):
    Yiabs = np.abs(Yi)
    SYi = np.power(Yiabs[:int(N/2)+1], 2)/N
    show_signal(SYi)
    return SYi

def power_spectral_density_function_of_the_noiseless_signal(SYi, SZ):
    SXi = SYi - SZ
    SXi[SXi<0] = 0
    return SXi

def spectral_substraction(Yi, Zi, alfa, beta):
    denominator = np.power(abs(Yi), beta)
    nominator = denominator - alfa*abs(Zi)
    nominator[nominator<0] = 0
    Xi = np.power(nominator/denominator, 1/beta)*Yi
    return Xi


def insert_frame(xe, xei, clear_noise_end, N, overlap, frames, i, padding_size):
    if i*(N-overlap)+N > len(xe):
        xe[i*(N-overlap): i*(N-overlap)+(N-padding_size)] += xei[:-padding_size]
    else:
        xe[i*(N-overlap): i*(N-overlap)+N] += xei
    return xe

def get_number_of_frames(Nx, N, overlap):
    frames = int(np.ceil(Nx/(N-overlap)))
    return frames


def get_frame(y, clear_noise_end, frames, N, i, overlap):
    padding_size = 0
    if clear_noise_end+i*(N-overlap)+N > len(y):
        yi = y[clear_noise_end+i*(N-overlap):]
        padding_size = N-len(yi)
        zeros = np.zeros(padding_size)
        yi = np.hstack((yi, zeros))
    else:
        yi = y[clear_noise_end+i*(N-overlap): clear_noise_end+i*(N-overlap)+N]
    return yi, padding_size


def generalized_spectral_density_estimation_of_the_noise(y, clear_noise_end, N, general, beta):
    z = y[:clear_noise_end]
    frames = int(clear_noise_end/N)

    if general:
        Z = np.zeros(N)
    else:
        Z = np.zeros(int(N/2)+1)

    for i in range(frames):
        zi = z[i*N: (i+1)*N]
        Zi = np.fft.fft(zi, N)
        Ziabs = np.abs(Zi)
        if general:
            Z += np.power(Ziabs, beta)
        else:
            Z += np.power(Ziabs[:int(N/2)+1], 2)/N

    Z = Z/frames                                 # V
    # show_signal(SZ)
    return Z

def create_denoising_filter(SXi, SYi, eps=1e-6):
    if eps is not None:
        SYi[SYi<eps] = eps
    Ail = np.sqrt(np.divide(SXi, SYi))
    Air = np.flip(Ail[1:-1], 0)
    Ai = np.hstack((Ail, Air))
    return Ai

def power_spectral_density_estimation_of_the_noisy_signal(Yi, N):
    Yiabs = np.abs(Yi)
    SYi = np.power(Yiabs[:int(N/2)+1], 2)/N
    # show_signal(SYi)
    return SYi


def power_spectral_density_function_of_the_noiseless_signal(SYi, SZ):
    SXi = SYi - SZ
    SXi[SXi<0] = 0
    return SXi

def generalized_spectral_substraction(y, clear_noise_end, N=512, general=True, overlap=257, alfa=1.5, beta=2):
    Zi = generalized_spectral_density_estimation_of_the_noise(y, clear_noise_end, N, general, beta)
    if general is False:
        SZ = Zi

    Nx = len(y)-clear_noise_end
    frames = get_number_of_frames(Nx, N, overlap)
    xe = np.zeros(Nx)

    for i in range(int(frames)):
        #print("frame:", i)
        yi, padding_size = get_frame(y, clear_noise_end, frames, N, i, overlap)
        Yi = np.fft.fft(yi, N)                                                      # v
        
        if general:
            Xi = spectral_substraction(Yi, Zi, alfa, beta)
            xei = np.fft.ifft(Xi, N).real                                           # v
            xwi = window(xei, N)
            xei = xwi
        else:
            SYi = power_spectral_density_estimation_of_the_noisy_signal(Yi, N)      
            SXi = power_spectral_density_function_of_the_noiseless_signal(SYi, SZ)
            Ai = create_denoising_filter(SXi, SYi)                                      # v
            Xi = evaluate_denoised_signal(Ai, Yi)                                       # v
            xei = np.fft.ifft(Xi, N).real                                               # v

        xe = insert_frame(xe, xei, clear_noise_end, N, overlap, frames, i, padding_size)
    return xe

def evaluate_denoised_signal(Ai, Yi):       # V
    Xi = Ai*Yi
    return Xi

def generalized_spectral_substraction_foyer(y, clear_noise_end, N=512, general=True, overlap=257, alfa=2, beta=2):
    if general is False:
        overlap = 0
        alfa = 0
        beta = 0

    if y.ndim > 1:
        # several channels
        h, w = np.transpose(y).shape
        xe = np.zeros((h, w-clear_noise_end))
        for i in range(y.ndim):
            xei = generalized_spectral_substraction(y[:, i], clear_noise_end, N, general, overlap, alfa, beta)
            xe[i,:] = xei
        xe = np.transpose(xe)
    else:
        # one channel
        xe = generalized_spectral_substraction(y, clear_noise_end, N, general, overlap, alfa, beta)
    
    fig = plt.figure(figsize=(8, 8))
    plt.plot(xe)
    fig.savefig("data/img/spectral_substraction_noisy.jpg")  # or you can pass a Figure object to pdf.savefig
    plt.close()
    return xe  
     
#quitarbajas(sound)
# START OF THE SCRIPT
if __name__ == "__main__":
    # Audio files paths
    sound= "data/audio/AVFAD/AAC/AAC002.wav"
    #sound= "data/audio/AVFAD/test/frank.vigil-75c6de05-0cc9-47cc-9b69-d4df79931f0e.m4a.wav"
    #sound= "data/audio/AVFAD/test/prueba.wav"
    
    input_path  = sound
    Vx, noisy_path = ruido(input_path)
    output_path = "filtered.wav"

    Fs, snd, t = load_audios(input_path)
    # Parameters
    N = 512                     # length of windows (odd for general method, even for non general)
    general = True              # choose if general or not general method should be used
    overlap = int((N+1)/2)      # overlap length (for general only)
    alfa = 8                    # parameter (for general only)
    beta = 2                    # parameter (for general only)

    quitarbajas(sound)
    ruido(sound)
    xe = generalized_spectral_substraction_foyer(snd, Fs, N, general, overlap, alfa, beta)
    
    show_signal(xe)
    save_audios("noisy.wav", xe, Fs)

    
  
