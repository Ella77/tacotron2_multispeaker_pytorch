# -*- coding: utf-8 -*
import os
import time
import pathlib
import librosa
from tqdm import tqdm
from shutil import copyfile
from scipy.io import wavfile
from multiprocessing import Pool
import io

#SR = 22050
SR = 24000
TOP_DB = 40


def main(output_directory, data):
    """
    Parse commandline arguments.
    data: list of tuples (source_directory, speaker_id, process_audio_flag)
    """

    jobs = []

    for source_directory, speaker_id, process_audio_flag in tqdm(data):
        for path, dirs, files in os.walk(source_directory):
            #if 'wavs' in dirs and 'metadata.txt' in files:
            if 'metadata.txt' in files:
                speaker_name = source_directory.split('/')[-1]

                sub_jobs = process(path, output_directory, speaker_name, speaker_id, process_audio_flag)

                jobs += sub_jobs

    print('Files to convert:', len(jobs))

    time.sleep(5)

    with Pool(42) as p:
        p.map(mapper, jobs)

    print('Done!')


def process(path, output_directory, speaker_name, speaker_id, process_audio=True):
    # print('---------------------------------------------------------------------------------')
    # print('path:', path)
    # print('speaker_name:', speaker_name)
    # print('file_prefix:', file_prefix)
    cmds = []

    with io.open(os.path.join(path, 'metadata.txt'), 'r',encoding='utf-8-sig') as file:
        lines = []
        files_to_process = []
        output_path = os.path.join(output_directory, speaker_name)
        output_audio_path = os.path.join(output_path, 'wavs')
        inter_audio_path = os.path.join(output_path, 'wavs_inter')

        pathlib.Path(output_audio_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(inter_audio_path).mkdir(parents=True, exist_ok=True)
        # print(path)
        for line in file:
            parts = line.strip().split('|')

            file_name = parts[0].split('/')[-1]
            text = parts[1]

            # if len(parts) == 3:
            #     text = parts[2]

            if not file_name.endswith('.wav'):
                file_name = file_name + '.wav'

            input_file_path = os.path.join(path, file_name)
            #input_file_path = os.path.join(path, 'wavs', file_name)
            inter_file_path = os.path.join(inter_audio_path, file_name)
            final_file_path = os.path.join(output_audio_path, file_name)

            files_to_process.append((input_file_path, inter_file_path, final_file_path, process_audio))

            new_line = '|'.join([final_file_path, text, str(speaker_id)]) + '\n'

            lines.append(new_line)

        with io.open(os.path.join(output_path, 'data.txt'), 'a+',encoding='utf-8-sig') as f:
            f.writelines(lines)




    return files_to_process


def mapper(job):
    try :
        fin, fint, fout, process_audio = job

        if process_audio:
            data, _ = librosa.load(fin, sr=SR)
            data, _ = librosa.effects.trim(data, top_db=TOP_DB)

            wavfile.write(fint, SR, data)

            command = "ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {}".format(fint, SR, fout)
            os.system(command)
        else:
            copyfile(fin, fout)
    except :
        print(fin)


if __name__ == '__main__':
    output_directory = './after_preprocessed_data'

    data = [
        (
            './data/boyoung_audio', #'/workspace/data/aws/dataset/samantha_fresh',
            0,
            True
        ),
        (
            './data/boyoung_audio_1.0down', #'/workspace/data/aws/dataset/samantha_fresh',
            1,
            True
        ),
        (
            './data/boyoung_audio_1.0up', #'/workspace/data/aws/dataset/samantha_fresh',
            2,
            True
        ),
        (
            './data/boyoung_audio_2.0down', #'/workspace/data/aws/dataset/samantha_fresh',
            3,
            True
        ),
        (
            './data/boyoung_audio_2.0up', #'/workspace/data/aws/dataset/samantha_fresh',
            4,
            True
        ),
        (
            './data/shinhye_audio', #'/workspace/data/aws/dataset/samantha_fresh',
            5,
            True
        ),
        (
            './data/shinhye_audio_1.0down', #'/workspace/data/aws/dataset/samantha_fresh',
            6,
            True
        ),
        (
            './data/shinhye_audio_1.0up', #'/workspace/data/aws/dataset/samantha_fresh',
            7,
            True
        ),
        (
            './data/shinhye_audio_2.0down', #'/workspace/data/aws/dataset/samantha_fresh',
            8,
            True
        ),
        (
            './data/shinhye_audio_2.0up', #'/workspace/data/aws/dataset/samantha_fresh',
            9,
            True
        ),
        (
            './data/kongyou_audio', #'/workspace/data/aws/dataset/samantha_fresh',
            10,
            True
        ),
        (
            './data/kongyou_audio_1.0down', #'/workspace/data/aws/dataset/samantha_fresh',
            11,
            True
        ),
        (
            './data/kongyou_audio_1.0up', #'/workspace/data/aws/dataset/samantha_fresh',
            12,
            True
        ),
        (
            './data/kongyou_audio_2.0down', #'/workspace/data/aws/dataset/samantha_fresh',
            13,
            True
        ),
        (
            './data/kongyou_audio_2.0up', #'/workspace/data/aws/dataset/samantha_fresh',
            14,
            True
        ),
        (
            './data/fv01', #'/workspace/data/aws/dataset/samantha_fresh',
            15,
            True
        ),   (
            './data/fv02', #'/workspace/data/aws/dataset/samantha_fresh',
            16,
            True
        ),   (
            './data/fv03', #'/workspace/data/aws/dataset/samantha_fresh',
            17,
            True
        ),   (
            './data/fv04', #'/workspace/data/aws/dataset/samantha_fresh',
            18,
            True
        ),
        (
            './data/fv05', #'/workspace/data/aws/dataset/samantha_fresh',
            19,
            True
        ),
        (
            './data/mv01', #'/workspace/data/aws/dataset/samantha_fresh',
            20,
            True
        ),
        (
            './data/mv02', #'/workspace/data/aws/dataset/samantha_fresh',
            21,
            True
        ),
        (
            './data/mv03', #'/workspace/data/aws/dataset/samantha_fresh',
            22,
            True
        ), (
            './data/mv04', #'/workspace/data/aws/dataset/samantha_fresh',
            23,
            True
        ),
        (
            './data/mv05', #'/workspace/data/aws/dataset/samantha_fresh',
            24,
            True
        ),











        # (
        #     '/workspace/data/aws/dataset/blizzard_2013',
        #     2,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_US/by_book/female/judy_bieber',
        #     3,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_US/by_book/female/mary_ann',
        #     4,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_UK/by_book/female/elizabeth_klett',
        #     5,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_US/by_book/male/elliot_miller',
        #     6,
        #     True
        # )
    ]

    main(output_directory, data)
