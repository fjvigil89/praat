import json, os
import sys, csv
import numpy as np
import pandas as pd
import Praat_favio as praat
import  shlex, subprocess
def export_csv():    
    row =["SPK_ID","Surgery","Visit Date","Visit Place","Age","Sex","Smoking","Job","Diagnostic","Details","Doctor","grbas-g","grbas-r","grbas-b","grbas-a","grbas-s","tfon","Cuestionary","Cuestionary Date","VHI_F","VHI_P","VHI_E","ECV Date","ECV Surgery","General health (F)",	"General Voice Aspects (P)","Voice Perception (E)",	"Sensations (S)","Intensity", "tone and timbre (T)","Environmental conditions (C)","psycho-emotional conditions (L)","f0_mean (Hz)","f0_std (Hz)","jitter_local (%)	","jitter_local_abs (s)", "jitter_rap (%)","jitter_ppq5 (%)	","jitter_ddp (%)","shimmer_local (%)",	"shimmer_local_dB (dB)","shimmer_apq3 (%)",	"shimmer_apq5 (%)",	"shimmer_apq11 (%)","shimmer_dda (%)", "nhr_mean"]    
    df = pd.DataFrame(columns=row)    
    
    id = []
    edad=[]
    surgery=[]
    visitDate=[]
    visitPlace=[]
    sex=[]
    Smoking=[]
    Job=[]
    Diagnostic=[]
    Details=[]
    Doctor=[]
    grbas_g=[]
    grbas_r=[]
    grbas_b=[]
    grbas_a=[]
    grbas_s=[]
    tfon=[]
    cuestionary=[]
    cuestionaryDate=[]
    VHI_F=[]
    VHI_P=[]
    VHI_E=[]
    ECVDate=[]
    ECVSurgery=[]
    Generalhealth=[]
    GeneralVoiceAspects=[]
    VoicePerception=[]
    Sensations=[]
    Intensity=[]
    toneandtimbre=[]
    Environmentalconditions=[]
    psychoemotionalconditions=[]
    f0_mean=[]
    f0_std=[]
    jitter_local=[]
    jitter_local_abs=[]
    jitter_rap=[]
    jitter_ppq5=[]
    jitter_ddp =[]
    shimmer_local =[]
    shimmer_local_dB=[]
    shimmer_apq3 =[]
    shimmer_apq5 =[]
    shimmer_apq11 =[]
    shimmer_dda=[]
    nhr_mean=[]
    data_path ="data/audio/dataset_TuVoz"
    for r, d, f in os.walk(data_path):     
        for d in d:
            path=data_path+"/"+d
            for r, d, f in os.walk(path):    
                for file in f:                    
                    if '.json' in file and (file.split('_')[1].split('.')[0]=="1" or file.split('_')[1].split('.')[0]=="2" or file.split('_')[1].split('.')[0]=="3"):  
                        try:
                            jsonfile = open(path+"/"+file,'r', encoding='utf-8')
                            jsonData = json.load(jsonfile) 
                            # print(jsonData)
                            
                            inputt= path+"/"+file.split('.')[0]+".wav"
                            output= "data/audio/tmp/"+file.split('.')[0]+".wav"
                            command_line = 'ffmpeg -y -i '+inputt+' '+output                        
                            args = shlex.split(command_line)
                            subprocess.call(args)
                            
                            id.append(file.split('.')[0])          
                            edad.append(jsonData['edad'])
                            surgery.append('PRE')
                            visitDate.append(jsonData['date'])
                            visitPlace.append('Consulta Hospital')
                            sex.append(jsonData['sexo'])
                            Smoking.append('null')
                            Job.append('null')
                            Diagnostic.append(jsonData['diagnostico'])
                            Details.append(jsonData['detalles'])
                            Doctor.append('null')
                            grbas_g.append(jsonData['grbas']['g'])
                            grbas_r.append(jsonData['grbas']['r'])
                            grbas_b.append(jsonData['grbas']['b'])
                            grbas_a.append(jsonData['grbas']['a'])
                            grbas_s.append(jsonData['grbas']['s'])
                            tfon.append(jsonData['tmf'])
                            
                            data_praat= praat.measure2Pitch_with_VAD(output, 75, 500, "Hertz")
                            
                            f0_mean.append(data_praat[0])
                            f0_std.append(data_praat[1])
                            nhr_mean.append(data_praat[2])
                            jitter_local.append(data_praat[3])
                            jitter_local_abs.append(data_praat[4])
                            jitter_rap.append(data_praat[5])
                            jitter_ppq5.append(data_praat[6])
                            jitter_ddp .append(data_praat[7])
                            shimmer_local .append(data_praat[8])
                            shimmer_local_dB.append(data_praat[9])
                            shimmer_apq3 .append(data_praat[10])
                            shimmer_apq5 .append(data_praat[11])
                            shimmer_apq11 .append(data_praat[12])
                            shimmer_dda.append(data_praat[13])
                        except:
                            f0_mean.append('null')
                            f0_std.append('null')
                            nhr_mean.append('null')
                            jitter_local.append('null')
                            jitter_local_abs.append('null')
                            jitter_rap.append('null')
                            jitter_ppq5.append('null')
                            jitter_ddp .append('null')
                            shimmer_local .append('null')
                            shimmer_local_dB.append('null')
                            shimmer_apq3 .append('null')
                            shimmer_apq5 .append('null')
                            shimmer_apq11 .append('null')
                            shimmer_dda.append('null')                            
                        
                        
    df['SPK_ID']=id
    df['Surgery']=surgery
    df['Visit Date']=visitDate
    df['Visit Place']=visitPlace
    df['Age']=edad
    df["Sex"]=sex
    df["Smoking"]=Smoking
    df["Job"]=Job
    df["Diagnostic"]=Diagnostic
    df["Details"]=Details
    df["Doctor"]=Doctor
    df["grbas-g"]=grbas_g
    df["grbas-r"]=grbas_r
    df["grbas-b"]=grbas_b
    df["grbas-a"]=grbas_a
    df["grbas-s"]=grbas_s
    df["tfon"]=tfon
    df["Cuestionary"]='null'
    df["Cuestionary Date"]='null'
    df["VHI_F"]='null'
    df["VHI_P"]='null'
    df["VHI_E"]='null'
    df["ECV Date"]='null'
    df["ECV Surgery"]='null'
    df["General health (F)"]='null'
    df["General Voice Aspects (P)"]='null'
    df["Voice Perception (E)"]='null'
    df["Sensations (S)"]='null'
    df["Intensity"]='null'
    df["tone and timbre (T)"]='null'
    df["Environmental conditions (C)"]='null'
    df["psycho-emotional conditions (L)"]='null'    
    df["f0_mean (Hz)"]=f0_mean
    df["f0_std (Hz)"]=f0_std
    df["jitter_local (%)	"]=jitter_local
    df["jitter_local_abs (s)"]=jitter_local_abs
    df["jitter_rap (%)"]=jitter_rap
    df["jitter_ppq5 (%)	"]=jitter_ppq5
    df["jitter_ddp (%)"]=jitter_ddp
    df["shimmer_local (%)"]=shimmer_local
    df["shimmer_local_dB (dB)"]=shimmer_local_dB
    df["shimmer_apq3 (%)"]=shimmer_apq3
    df["shimmer_apq5 (%)"]=shimmer_apq5
    df["shimmer_apq11 (%)"]=shimmer_apq11
    df["shimmer_dda (%)"]=shimmer_dda
    df["nhr_mean"]=nhr_mean
    

    df.to_excel("data/xlsx/dataset_tuvoz_vocal.xlsx")
   
if __name__ == '__main__':                  
    data2 = export_csv() 
    #print(data2)