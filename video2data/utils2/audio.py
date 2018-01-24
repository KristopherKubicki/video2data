#!/usr/bin/python3 
# coding: utf-8
#

#
# given a segment of audio, determine its energy
#  check if that energy is in the human band of 
#  speech.  
#
#  warning - this has a defect in it that will crash for long
#  segments of audio.  Needs to be fixed
#
def speech_calc(self,raw_audio):
  if len(raw_audio) < 10:
    return
  data_freq = np.fft.fftfreq(len(raw_audio),1.0/SAMPLE)
  data_freq = data_freq[1:]
  data_ampl = np.abs(np.fft.fft(raw_audio))
  data_ampl = data_ampl[1:]
  data_energy = data_ampl ** 2

  energy_freq = {}
  for (i, freq) in enumerate(data_freq):
    if abs(freq) not in energy_freq:
      energy_freq[abs(freq)] = data_energy[i] * 2

  sum_energy = 0
  for f in energy_freq.keys():
    if 300<f<3000:
      sum_energy += energy_freq[f]

  # sometimes I get errors here
  return sum_energy / sum(energy_freq.values())

