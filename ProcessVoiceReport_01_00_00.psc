# Process Voice Report
# May-June 2015
# Luis M. T. Jesus, University of Aveiro, Portugal
# lmtj@ua.pt
# http://sweet.ua.pt/lmtj/

form ProcessFiles
   text file_extension .prt
   text input_directory /home/frank/sites/praat/data/audio/AVFAD/
endform

Read Strings from raw text file... ListFiles.txt
Genericize
Sort
numberOfFiles = Get number of strings

file_out$ = "results.txt"
filedelete 'file_out$'

for ifile to numberOfFiles
	select Strings ListFiles
	participant$ = Get string... ifile
	Read from file... 'input_directory$''participant$'/'participant$'002'file_extension$'
	select TextGrid 'participant$'002
	analysisStart = Get starting point... 1 2
	analysisEnd = Get end point... 1 2
	zoomEnd = analysisStart + 0.075
	phone$ = Get label of interval... 1 2
	select Sound 'participant$'002
	samplingFrequency = Get sampling frequency
	
	pitch = To Pitch (cc)... 0 75 15 no 0.03 0.45 0.15 0.35 0.14 500
	# Time step (s): 0; Pitch floor (Hz): 75; Max. number of candidates: 15; Very accurate: no; Silence threshold: 0.03; Voicing threshold: 0.45; Octave cost: 0.15; Octave-jump cost: 0.35; Voiced/unvoiced cost: 0.14; Pitch ceiling (Hz): 500
	plus Sound 'participant$'002
	To PointProcess (cc)
	plus Sound 'participant$'002
	View & Edit
	editor: "PointProcess " + participant$ + "002_" + participant$ + "002"
	Zoom... analysisStart zoomEnd
	endeditor
	pause
	plus pitch
	
	voiceReport$ = Voice report... analysisStart analysisEnd 75 500 1.3 1.6 0.03 0.45
	# Time range (s): analysisStart - analysisEnd ; Pitch range (Hz): 75 - 500; Maximum period factor: 1.3; Maximum amplitude factor: 1.6; Silence threshold: 0.03; Voicing threshold: 0.45
	voiceReport$ > Voice_Report.txt
	pulsesNumber = extractNumber(voiceReport$, "Number of pulses: ")
	unvoicedFrames = extractNumber(voiceReport$, "Fraction of locally unvoiced frames: ")
	numberVoiceBreaks = extractNumber(voiceReport$, "Number of voice breaks: ")
	degreeVoiceBreaks = extractNumber(voiceReport$, "Degree of voice breaks: ")
	f0_median = extractNumber(voiceReport$, "Median pitch: ")
	f0_mean = extractNumber(voiceReport$, "Mean pitch: ")
	f0_std = extractNumber(voiceReport$, "Standard deviation: ")
	f0_min = extractNumber(voiceReport$, "Minimum pitch: ")
	f0_max = extractNumber(voiceReport$, "Maximum pitch: ")
	jitter_local = 100*extractNumber(voiceReport$, "Jitter (local): ")
	jitter_local_abs = extractNumber(voiceReport$, "Jitter (local, absolute): ")
	jitter_rap = 100*extractNumber(voiceReport$, "Jitter (rap): ")
	jitter_ppq5 = 100*extractNumber(voiceReport$, "Jitter (ppq5): ")
	jitter_ddp = 100*extractNumber(voiceReport$, "Jitter (ddp): ")
	shimmer_local = 100*extractNumber(voiceReport$, "Shimmer (local): ")
	shimmer_local_dB = extractNumber(voiceReport$, "Shimmer (local, dB): ")
	shimmer_apq3 = 100*extractNumber(voiceReport$, "Shimmer (apq3): ")
	shimmer_apq5 = 100*extractNumber(voiceReport$, "Shimmer (apq5): ")
	shimmer_apq11 = 100*extractNumber(voiceReport$, "Shimmer (apq11): ")
	shimmer_dda = 100*extractNumber(voiceReport$, "Shimmer (dda): ")
	autocorrelation_mean = extractNumber(voiceReport$, "Mean autocorrelation: ")
	nhr_mean = extractNumber(voiceReport$, "Mean noise-to-harmonics ratio: ")
	hnr_mean = extractNumber(voiceReport$, "Mean harmonics-to-noise ratio: ")
	fileappend 'file_out$' 'participant$' 'samplingFrequency' 'phone$' 'pulsesNumber'
	... 'unvoicedFrames' 'numberVoiceBreaks' 'degreeVoiceBreaks'
	... 'f0_median' 'f0_mean' 'f0_std' 'f0_min' 'f0_max' 'jitter_local' 'jitter_local_abs' 'jitter_rap'
	... 'jitter_ppq5' 'jitter_ddp' 'shimmer_local' 'shimmer_local_dB' 'shimmer_apq3'
	... 'shimmer_apq5' 'shimmer_apq11' 'shimmer_dda' 'autocorrelation_mean' 'nhr_mean'
	... 'hnr_mean''newline$'
	select Sound 'participant$'002
	plus TextGrid 'participant$'002
	plus Pitch 'participant$'002
	plus PointProcess 'participant$'002_'participant$'002
	Remove
endfor
