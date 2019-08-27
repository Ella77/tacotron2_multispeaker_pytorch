# -*- coding: utf-8 -*
import os
import sys
import audioread
import pickle

from multiprocessing import Pool
from tqdm import tqdm
import io


sys.path.append('..')
sys.path.append('home/ubuntu/xvdf1/tacotron2_markornot/tacotron2_multispeaker_pytorch/processing')

# #
# sys.path.append()
from text import text_to_sequence

speakers = [
    'mv01',
    'mv02',
    'mv03',
    'mv04',
    'mv05',
    'fv01',
    'fv02',
    'fv03',
    'fv04',
    'fv05',
    'kongyou_audio',
    'kongyou_audio_1.0up',
    'kongyou_audio_1.0down',
    'kongyou_audio_2.0up',
    'kongyou_audio_2.0down',
    'shinhye_audio',
    'shinhye_audio_1.0up',
    'shinhye_audio_1.0down',
    'shinhye_audio_2.0up',
    'shinhye_audio_2.0down',
    'boyoung_audio',
    'boyoung_audio_1.0up',
    'boyoung_audio_1.0down',
    'boyoung_audio_2.0up',
    'boyoung_audio_2.0down'
]

#data_path = '/workspace/training_data/'
data_path = 'processing/after_preprocessed_data'

def mapper(line):
    fp, text, _ = line.strip().split('|')

    seq = text_to_sequence(text, ['korean_cleaners'])
    #print(seq)
    #print(seq.encode('UTF-8'))

    # if os.path.isfile(fp):
    #     with audioread.audio_open(fp) as f:
    #         duration = f.duration
    # else:
    #     duration = None
    #

    try :
        with audioread.audio_open(os.path.join(data_path,fp.split('/')[2],fp.split('/')[3],fp.split('/')[4])) as f:
            duration = f.duration
    except :
        duration  = None
        print("nofile", fp)



    return fp, len(seq), duration

'''
python processing/check_distributions.py 
'''
if __name__ == '__main__':
    data = {}
    for sp in tqdm(speakers):
        data[sp] = []

        print(sp)
        print(os.getcwd())
        #print(sys.path)
        with io.open(os.path.join(data_path, sp, 'data.txt'), 'r',encoding='utf-8-sig') as f:
            lines = [l for l in f]

        with Pool(64) as p:
            result = p.map(mapper, lines)

        data[sp] = result

    with open('data.pickles', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done!')
