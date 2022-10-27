import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

path_destinity="data/txt/"
sound= "data/audio/AVFAD/AAC/AAC002.wav"

def TF(sound):
    sampling_freq, audio = wavfile.read(sound, 'r')   # Leer archivo

    audio = audio / np.max(audio)   #  Normalización

    #  Aplicar transformada de Fourier
    fft_signal = np.fft.fft(audio)
    print(fft_signal)
    # [-0.04022912+0.j         -0.04068997-0.00052721j -0.03933007-0.00448355j
    #  ... -0.03947908+0.00298096j -0.03933007+0.00448355j -0.04068997+0.00052721j]

    fft_signal = abs(fft_signal)
    print(fft_signal)
    # [0.04022912 0.04069339 0.0395848  ... 0.08001755 0.09203427 0.12889393]

    #  Crear línea de tiempo
    Freq = np.arange(0, len(fft_signal))

    #  Dibujo de señales de habla
    plt.figure()
    plt.plot(Freq, fft_signal, color='blue')
    plt.xlabel('Freq (in kHz)')
    plt.ylabel('Amplitude')
    plt.show()

TF(sound)