from statistics import median
import parselmouth
import json
import numpy as np
import Energy_VAD as vad
import matplotlib.pyplot as plt
import pandas as pd
from parselmouth.praat import call
import warnings
import soundfile
import scipy.io.wavfile as waves

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", np.ComplexWarning)
import scipy.io.wavfile as waves


def find_intervalo(sx_len, porciento):
    if sx_len > 3:
        start = (sx_len*porciento)/100
        end = (sx_len*(100-porciento))/100
    else:
        start = 0
        end = sx_len

    return start, end

def measurePitch(voiceID, f0min= 60, f0max=500, unit="Hertz"):
    sound = parselmouth.Sound(voiceID)

    start_time = 3.413321
    end_time = 4.560055

    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)

    pulses = call([sound, pitch], "To PointProcess (cc)")
    duration = call(sound, "Get total duration")  # duration
    voice_report_str = call([sound, pitch, pulses], "Voice report", start_time, end_time, 75, 500, 1.3, 1.6, 0.03,
                            0.45)

    return voice_report_str

def measure2Pitch(voiceID, f0min= 60, f0max=500, unit="Hertz"):

    sound = parselmouth.Sound(voiceID)  # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)

    pulses = call([sound, pitch], "To PointProcess (cc)")

    duration = call(sound, "Get total duration")  # duration
    start_time, end_time = find_intervalo(duration, duration, 0.4)
    print("Intervalo de voz: " + str(start_time) + " a " + str(end_time))

    # "Voice report"
    meanF0 = (round(call(pitch, "Get mean", start_time, end_time, unit), 3))  # get mean pitch
    stdevF0 = (
        round(call(pitch, "Get standard deviation", start_time, end_time, unit), 3))  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.03, 1.0)
    hnr = (call(harmonicity, "Get mean", start_time, end_time))

    period_floor = 0.0001  # Intervalo más pequeño (seg)
    period_ceiling = 0.03  # Intervalo más grande (seg)
    max_period = 1.3  # Diferencia más grande entre intervalos consecutivos
    max_amp = 1.6  # Valor de máxima amplitud

    localJitter = (
        round(100 * call(pulses, "Get jitter (local)", start_time, end_time, 0.0001, period_ceiling, 1.3), 3))
    localabsoluteJitter = (
        call(pulses, "Get jitter (local, absolute)", start_time, end_time, 0.0001, period_ceiling, 1.3))
    rapJitter = (
        round(100 * call(pulses, "Get jitter (rap)", start_time, end_time, 0.0001, period_ceiling, 1.3), 3))
    ppq5Jitter = (
        round(100 * call(pulses, "Get jitter (ppq5)", start_time, end_time, 0.0001, period_ceiling, 1.3), 3))
    ddpJitter = (
        round(100 * call(pulses, "Get jitter (ddp)", start_time, end_time, 0.0001, period_ceiling, 1.3), 3))
    localShimmer = (
        round(100 * call([sound, pulses], "Get shimmer (local)", start_time, end_time, 0.0001, period_ceiling, 1.3,
                            1.6), 3))
    localdbShimmer = (round(
        call([sound, pulses], "Get shimmer (local_dB)", start_time, end_time, 0.0001, period_ceiling, 1.3, 1.6), 3))
    apq3Shimmer = (round(
        100 * call([sound, pulses], "Get shimmer (apq3)", start_time, end_time, 0.0001, period_ceiling, 1.3, 1.6),
        3))
    aqpq5Shimmer = (round(
        100 * call([sound, pulses], "Get shimmer (apq5)", start_time, end_time, 0.0001, period_ceiling, 1.3, 1.6),
        3))
    apq11Shimmer = (
        round(100 * call([sound, pulses], "Get shimmer (apq11)", start_time, end_time, 0.0001, period_ceiling, 1.3,
                            1.6), 3))
    ddaShimmer = (round(
        100 * call([sound, pulses], "Get shimmer (dda)", start_time, end_time, 0.0001, period_ceiling, 1.3, 1.6),
        3))

    columns = ["meanF0", "stdev", " hnr", " localJitter", " localabsoluteJitter", " rapJitter", " ppq5Jitter",
                " ddpJitter", " localShimmer", " localdbShimmer", " apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
                "ddaShimmer"]
    row = [meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer,
            localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]

    return json.dumps(columns), json.dumps(row)

def measure2Pitch_with_VAD(voiceID, f0min= 60, f0max=500, unit="Hertz"):

    path_sx_with_vad = vad.energia(voiceID, 0.025, 0.01)
    sound = parselmouth.Sound(path_sx_with_vad)

    duration = call(sound, "Get total duration")  # duration
    start_time, end_time = find_intervalo(duration, 5)
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.15, 0.35, 0.14, f0max)
    pulses = call([sound, pitch], "To PointProcess (cc)")

    meanF0 = call(pitch, "Get mean", start_time, end_time, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", start_time, end_time, unit)  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.03, 1.0)
    hnr = call(harmonicity, "Get mean", start_time, end_time)

    harmonicity_gne = sound.to_harmonicity_gne(minimum_frequency=f0min, maximum_frequency=f0max, bandwidth=(f0max - f0min), step=(f0max - f0min))
    gne_mean = np.mean(harmonicity_gne)
    gne_std = np.std(harmonicity_gne)

    period_floor = 0.0001  # Intervalo más pequeño (seg)
    period_ceiling = 0.03  # Intervalo más grande (seg)
    max_period = 1.3  # Diferencia más grande entre intervalos consecutivos
    max_amp = 1.6  # Valor de máxima amplitud

    localJitter = 100 * call(pulses, "Get jitter (local)", start_time, end_time, 0.0001, period_ceiling, 1.3)
    localabsoluteJitter = call(pulses, "Get jitter (local, absolute)", start_time, end_time, 0.0001, period_ceiling,
                                1.3)
    rapJitter = 100 * call(pulses, "Get jitter (rap)", start_time, end_time, 0.0001, period_ceiling, 1.3)
    ppq5Jitter = 100 * call(pulses, "Get jitter (ppq5)", start_time, end_time, 0.0001, period_ceiling, 1.3)
    ddpJitter = 100 * call(pulses, "Get jitter (ddp)", start_time, end_time, 0.0001, period_ceiling, 1.3)
    localShimmer = 100 * call([sound, pulses], "Get shimmer (local)", start_time, end_time, 0.0001, period_ceiling,
                                1.3, 1.6)
    localdbShimmer = call([sound, pulses], "Get shimmer (local_dB)", start_time, end_time, 0.0001, period_ceiling,
                            1.3, 1.6)
    apq3Shimmer = 100 * call([sound, pulses], "Get shimmer (apq3)", start_time, end_time, 0.0001, period_ceiling,
                                1.3, 1.6)

    aqpq5Shimmer = 100 * call([sound, pulses], "Get shimmer (apq5)", start_time, end_time, 0.0001, period_ceiling,
                                1.3, 1.6)
    apq11Shimmer = 100 * call([sound, pulses], "Get shimmer (apq11)", start_time, end_time, 0.0001, period_ceiling,
                                1.3, 1.6)
    ddaShimmer = 100 * call([sound, pulses], "Get shimmer (dda)", start_time, end_time, 0.0001, period_ceiling,
                            1.3, 1.6)

    # Unvoiced ratio compute
    count_ceros = 0
    for i in pitch.selected_array:
        if i[0] == 0:
            count_ceros += 1

    UVR = count_ceros / len(pitch.selected_array)

    columns = ["meanF0", "stdev", " hnr", " localJitter", " localabsoluteJitter", " rapJitter", " ppq5Jitter",
                " ddpJitter", " localShimmer", " localdbShimmer", " apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
                "ddaShimmer", "mean GNE", "std GNE", "UVRatio"]
    row = [meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer,
            localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, gne_mean, gne_std, UVR]

    return json.dumps(columns), json.dumps(row)


# if __name__ == '__main__':
#     """ args = sys.argv[1:]        
#     n =int(args[0]) 
#     sound = args[1]
# """

#     sound = "data/audio/AVFAD/AAF/AAF002.wav"
#     data2 = measure2Pitch_with_VAD(sound, 60, 500, "Hertz")
#     name_column = data2[0].split(',')
#     value_column = data2[1].split(',')
#     df = pd.DataFrame(
#         {'Parámetros': name_column,
#          "Señal AV-FAD": value_column,
#          })
#     df.head()
#     print(df)
