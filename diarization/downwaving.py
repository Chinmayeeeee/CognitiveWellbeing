import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Audio
import soundfile as sf

y, s = librosa.load('speakers/audio/S0-1.wav', sr=16000)
# y, s = librosa.load('speakers/S0-0.wav')
print(y, s)
sf.write('speakers/audio/S0-1.wav', y, s)

data, samplerate = sf.read('speakers/audio/S0-1.wav')
print(data, samplerate)


