import scipy.io.wavfile as wf

def load_wav_file(path):
    a, b = wf.read(path)
    return a, b
