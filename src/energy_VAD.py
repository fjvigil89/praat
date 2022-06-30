import numpy as np
import scipy.io.wavfile as waves
import soundfile as sf
from sklearn.mixture import GaussianMixture

import warnings

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)


def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a[0], b[0], c[0]])


def windowing(x, fs=16000, Ns=0.025, Ms=0.010):  # tipo rectángulo

    N = int(Ns * fs)
    M = int(Ms * fs)
    T = len(x)
    m = np.arange(0, T - N, M).reshape(1, -1)
    L = len(m)
    ind = np.arange(N).reshape(-1, 1).dot(np.ones((1, L))) + np.ones((N, 1)).dot(m)

    return x[ind.astype(int).T].astype(np.float32)


def energia(sound, Ns=0.025, Ms=0.010):
    # Load signal
    muestreo, snd = waves.read(sound)

    # # normalizar la señal del wav
    snd = snd / (2. ** 15)

    # cálculo de la energía
    tramas = windowing(snd, muestreo)
    eng_sum = np.sum(tramas ** 2, axis=1)
    #eng_sum = 10*np.log10(eng_sum)  # Convertir a dB

    
    
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

    delta = np.diff(relacion)
    if not relacion[0]:
        delta = np.argwhere(delta == True)
    else:
        delta = np.argwhere(delta == False)

    if len(delta) % 2 != 0:
        np.append(delta, len(eng_sum) - 1)

    # Macheo de la señal
    salida = []
    if len(delta) == 1:
        salida = snd
    else:
        for i in range(0, len(delta) - 1, 2):
            init = int((delta[i] * Ms) * muestreo)
            end = int((delta[i + 1] * Ms) * muestreo)
            salida.append(snd[init: end])
        salida = np.concatenate(salida)

    sf.write('senal_salida_umbral.wav', salida, muestreo, 'PCM_16')
    return 'senal_salida_umbral.wav'

