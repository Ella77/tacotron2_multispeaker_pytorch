import numpy as np
import torch
from scipy.io.wavfile import write
import models
from tacotron2.text import text_to_sequence
import matplotlib.pyplot as plt
import sys

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
        fig.savefig('data'+str(i)+'.png')
    plt.close(fig)
sys.path.append('./waveglow')

tacotron_path = 'output/checkpoint_Tacotron2_18'
taco_checkpoint = torch.load(tacotron_path, map_location='cpu')
state_dict = torch.load(tacotron_path)['state_dict']
t2 = models.get_model('Tacotron2', taco_checkpoint['config'], to_cuda=True)

text = "안녕하세요 박신혜입니다"
# preprocessing
inputs = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
print(inputs)
inputs = torch.from_numpy(inputs).to(device='cuda', dtype=torch.int64)
#inputs = torch.from_numpy(np.array([bombom, kitkat], dtype=np.int64)).to(device='cuda', dtype=torch.int64)

#input_lengths = torch.IntTensor([inputs.size(1), inputs.size(1)]).cuda().long()
input_lengths = torch.IntTensor([inputs.size(1)]).cuda().long()
speaker_id = torch.IntTensor([5]).cuda().long()
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
    audio = waveglow.infer(mel,sigma=0.6)
    print((audio).min().item(),(audio).max().item())

# audio = waveglow.infer(mel, sigma=sigma)
# if denoiser_strength > 0:
#     audio = denoiser(audio, denoiser_strength)
audio = audio * 32678.0
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


write("audioshinhye.wav", rate, audio)