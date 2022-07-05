import pandas as pd
import Praat as praat

if __name__ == '__main__':
    """ args = sys.argv[1:]        
    sound = args[0]
"""

    #sound = "data/audio/AVFAD/AAF/AAF002.wav"
    sound = "data/audio/tmp/TVD-T-0001_1.wav"
    data2 = praat.measure2Pitch_with_VAD(sound, 60, 500, "Hertz")
    name_column = data2[0].split(',')
    value_column = data2[1].split(',')
    df = pd.DataFrame(
        {'Parámetros': name_column,
            "Señal AV-FAD": value_column,
            })
    df.head()
    print(df)