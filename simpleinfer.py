import numpy as np
import torch
from scipy.io.wavfile import write
import models
from tacotron2.text import text_to_sequence
import matplotlib.pyplot as plt
import sys
from scipy import signal
import librosa

def inv_mel_spectrogram(mel_spectrogram):
    # Converts mel spectrogram to waveform using librosa
    D = _denormalize(mel_spectrogram)
    S = _mel_to_linear(_db_to_amp(D + 20))  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** 1.5),0.97)

def _build_mel_basis():
    #assert hparams.fmax <= hparams.sample_rate // 2

    #fmin: Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    #fmax: 7600, To be increased/reduced depending on data.
    #return librosa.filters.mel(hparams.sample_rate, hparams.fft_size, n_mels=hparams.num_mels,fmin=hparams.fmin, fmax=hparams.fmax)
    return librosa.filters.mel(24000, 2048, n_mels=80)  # fmin=0, fmax= sample_rate/2.0

def _mel_to_linear(mel_spectrogram):

    _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))
def _stft(y):


    return librosa.stft(y=y, n_fft=2048, hop_length=300, win_length=1200)

def _istft(y):
    return librosa.istft(y, hop_length=300, win_length=1200)


def _griffin_lim(S):
    # librosa implementation of Griffin-Lim
    # Based on https://github.com/librosa/librosa/issues/434

    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(60):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y
def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _denormalize(D):

    return (((np.clip(D, -4,
                      4) + 4) * -(-100) / (2 * 4))
            + (-100))
#         else:
#             return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
#
#     if hparams.symmetric_mels:
#         return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
#     else:
#         return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
def inv_preemphasis(wav, k):

    return signal.lfilter([1], [1, -k], wav)

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
        fig.savefig('data'+str(i)+'.png')
    plt.close(fig)
sys.path.append('./waveglow')

tacotron_path = 'output/checkpoint_Tacotron2_30'
taco_checkpoint = torch.load(tacotron_path, map_location='cpu')
state_dict = torch.load(tacotron_path)['state_dict']
t2 = models.get_model('Tacotron2', taco_checkpoint['config'], to_cuda=True)

text = "아들 진수가 살아 돌아온다"
# preprocessing
inputs = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
print(inputs)
inputs = torch.from_numpy(inputs).to(device='cuda', dtype=torch.int64)
#inputs = torch.from_numpy(np.array([bombom, kitkat], dtype=np.int64)).to(device='cuda', dtype=torch.int64)

#input_lengths = torch.IntTensor([inputs.size(1), inputs.size(1)]).cuda().long()
input_lengths = torch.IntTensor([inputs.size(1)]).cuda().long()
speaker_id = torch.IntTensor([0]).cuda().long()
embedded_speaker = t2.speakers_embedding(speaker_id)
print("speaker",embedded_speaker)



t2.load_state_dict(state_dict)
_ = t2.cuda().eval().half()

waveglow = torch.load('output/waveglow_128000')['model']
for m in waveglow.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')

waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval().half()
print("load waveglow")
# waveglow = waveglow.half()
for k in waveglow.convinv:
    k.float()
# from apex import amp
# waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")


# run the models
with torch.no_grad():
    print(speaker_id)
    mel_outputs, mel_outputs_postnet, _, alignments= t2.infer(inputs, speaker_id)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    mel = mel_outputs_postnet


    print(mel.shape)
    print("tacoinfered")
    # #griff-lim

    # audio = inv_mel_spectrogram(mel.cpu().numpy()[0])
    # #audio *= 32678.0
    # #
    # print((audio).min().item(),(audio).max().item())
    # write('grifffv01_6_epoch26_진수.wav', 24000, audio)


    audio = waveglow.infer(mel,sigma=0.6)
    print((audio).min().item(),(audio).max().item())

# audio = waveglow.infer(mel, sigma=sigma)
# if denoiser_strength > 0:
#     audio = denoiser(audio, denoiser_strength)
audio = audio * 32768.0
print((audio).min().item(),(audio).max().item())
audio = audio.squeeze()
audio = audio.cpu().numpy()
audio = audio.astype('int16')
# audio_path = os.path.join(
#     output_dir, "{}_synthesis.wav".format(file_name))
# write(audio_path, sampling_rate, audio)
plt.imshow(mel.squeeze(0).detach().cpu().numpy())
# audio_numpy = audio[0].data.cpu().numpy()
# min = np.amin(audio_numpy)
# max = np.amax(audio_numpy)
# print(min,max)
rate = 24000

#
write("fv01_epoch30_아.wav", rate, audio)
#


#

# m  = np.load('tacomol_02.0024'+'.npz')
# #m  = np.load('mv04_t05_s13'+'.npz')
# mel = m['mel']
# print(mel.shape)
# #mel = mel.transpose(1,0)
# print(mel.shape)
# audio = inv_mel_spectrogram(mel)
# write('sampleksss.wav', 24000, audio)
