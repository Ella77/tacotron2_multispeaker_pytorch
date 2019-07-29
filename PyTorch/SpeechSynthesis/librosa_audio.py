import os
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

wavs_in_path = "/workspace/data/aws/dataset/samantha/wavs_in"
wavs_out_path = "/workspace/data/aws/dataset/samantha/wavs"

sr = 22050
maxv = np.iinfo(np.int16).max

wavs = os.listdir(wavs_in_path)

for wav in tqdm(wavs):
    data, _ = librosa.load(os.path.join(wavs_in_path, wav), sr=sr)

    wavfile.write(os.path.join(wavs_out_path, wav), sr, (data*maxv).astype(np.int16))
