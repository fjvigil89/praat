'''
Prosody parameters
'''

import parselmouth

from parselmouth.praat import call
class Praat:    
    path_destinity="data/txt/"
    """ f0min=0.01
    f0max=600
    unit="Hertz" """

    # This is the function to measure voice pitch
    def measurePitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID) # read the sound

        #pitch = sound.to_pitch()
        pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.015, 0.35, 0.14, f0max)
        
        pulses = call([sound, pitch], "To PointProcess (cc)")
        duration = call(sound, "Get total duration") # duration
        voice_report_str = call([sound, pitch, pulses], "Voice report", 0.000001, duration, 75, 500, 1.3, 1.6, 0.03, 0.45)       
        
        return voice_report_str
    
    sound= "data/audio/AVFAD/AAO/AAO001.wav"
    data = measurePitch(sound, 75, 500, "Hertz")
    file=open(path_destinity+sound.split("/")[4].split(".")[0]+".txt","w")
    file.write(data)
    file.close()
    print(data)
