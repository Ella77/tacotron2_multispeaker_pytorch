import urllib.request
import torch
import os
import sys

# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


dependencies = ['torch']


def nvidia_ncf(pretrained=True, **kwargs):
    """Constructs an NCF model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool, True): If True, returns a model pretrained on ml-20m dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
        nb_users (int): number of users
        nb_items (int): number of items
        mf_dim (int, 64): dimension of latent space in matrix factorization
        mlp_layer_sizes (list, [256,256,128,64]): sizes of layers of multi-layer-perceptron
        dropout (float, 0.5): dropout
    """

    from PyTorch.Recommendation.NCF import neumf as ncf

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    config = {'nb_users': None, 'nb_items': None, 'mf_dim': 64, 'mf_reg': 0.,
              'mlp_layer_sizes': [256, 256, 128, 64], 'mlp_layer_regs':[0, 0, 0, 0], 'dropout': 0.5}

    if pretrained:
        if fp16:
            checkpoint = 'https://developer.nvidia.com/joc-ncf-fp16-pyt-20190225'
        else:
            checkpoint = 'https://developer.nvidia.com/joc-ncf-fp32-pyt-20190225'
        ckpt_file = os.path.basename(checkpoint)
        if not os.path.exists(ckpt_file) or force_reload:
            sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
            urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)

        if checkpoint_from_distributed(ckpt):
            ckpt = unwrap_distributed(ckpt)

        config['nb_users'] = ckpt['mf_user_embed.weight'].shape[0]
        config['nb_items'] = ckpt['mf_item_embed.weight'].shape[0]
        config['mf_dim'] = ckpt['mf_item_embed.weight'].shape[1]
        mlp_shapes = [ckpt[k].shape for k in ckpt.keys() if 'mlp' in k and 'weight' in k and 'embed' not in k]
        config['mlp_layer_sizes'] = [mlp_shapes[0][1], mlp_shapes[1][1], mlp_shapes[2][1],  mlp_shapes[2][0]]
        config['mlp_layer_regs'] = [0] * len(config['mlp_layer_sizes'])

    else:
        if 'nb_users' not in kwargs:
            raise ValueError("Missing 'nb_users' argument.")
        if 'nb_items' not in kwargs:
            raise ValueError("Missing 'nb_items' argument.")
        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v
        config['mlp_layer_regs'] = [0] * len(config['mlp_layer_sizes'])

    m = ncf.NeuMF(**config)

    if fp16:
        m.half()

    if pretrained:
        m.load_state_dict(ckpt)

    return m


def nvidia_tacotron2(pretrained=True, **kwargs):
    """Constructs a Tacotron 2 model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args (type[, default value]):
        pretrained (bool, True): If True, returns a model pretrained on LJ Speech dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
        n_symbols (int, 148): Number of symbols used in a sequence passed to the prenet, see
                              https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/text/symbols.py
        p_attention_dropout (float, 0.1): dropout probability on attention LSTM (1st LSTM layer in decoder)
        p_decoder_dropout (float, 0.1): dropout probability on decoder LSTM (2nd LSTM layer in decoder)
        max_decoder_steps (int, 1000): maximum number of generated mel spectrograms during inference
    """

    from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import model as tacotron2
    from PyTorch.SpeechSynthesis.Tacotron2.models import lstmcell_to_float, batchnorm_to_float
    from PyTorch.SpeechSynthesis.Tacotron2.tacotron2.text import text_to_sequence

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    if pretrained:
        if fp16:
            checkpoint = 'https://developer.nvidia.com/joc-tacotron2-fp16-pyt-20190306'
        else:
            checkpoint = 'https://developer.nvidia.com/joc-tacotron2-fp32-pyt-20190306'
        ckpt_file = os.path.basename(checkpoint)
        if not os.path.exists(ckpt_file) or force_reload:
            sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
            urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        state_dict = ckpt['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)
        config = ckpt['config']
    else:
        config = {'mask_padding': False, 'n_mel_channels': 80, 'n_symbols': 148,
                  'symbols_embedding_dim': 512, 'encoder_kernel_size': 5,
                  'encoder_n_convolutions': 3, 'encoder_embedding_dim': 512,
                  'attention_rnn_dim': 1024, 'attention_dim': 128,
                  'attention_location_n_filters': 32,
                  'attention_location_kernel_size': 31, 'n_frames_per_step': 1,
                  'decoder_rnn_dim': 1024, 'prenet_dim': 256,
                  'max_decoder_steps': 1000, 'gate_threshold': 0.5,
                  'p_attention_dropout': 0.1, 'p_decoder_dropout': 0.1,
                  'postnet_embedding_dim': 512, 'postnet_kernel_size': 5,
                  'postnet_n_convolutions': 5, 'decoder_no_early_stopping': False}
        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v

    m = tacotron2.Tacotron2(**config)

    if fp16:
        m = batchnorm_to_float(m.half())
        m = lstmcell_to_float(m)

    if pretrained:
        m.load_state_dict(state_dict)

    m.text_to_sequence = text_to_sequence

    return m


def nvidia_waveglow(pretrained=True, **kwargs):
    """Constructs a WaveGlow model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool): If True, returns a model pretrained on LJ Speech dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
    """

    from PyTorch.SpeechSynthesis.Tacotron2.waveglow import model as waveglow
    from PyTorch.SpeechSynthesis.Tacotron2.models import batchnorm_to_float

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    if pretrained:
        if fp16:
            checkpoint = 'https://developer.nvidia.com/joc-waveglow-fp16-pyt-20190306'
        else:
            checkpoint = 'https://developer.nvidia.com/joc-waveglow-fp32-pyt-20190306'
        ckpt_file = os.path.basename(checkpoint)
        if not os.path.exists(ckpt_file) or force_reload:
            sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
            urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        state_dict = ckpt['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)
        config = ckpt['config']
    else:
        config = {'n_mel_channels': 80, 'n_flows': 12, 'n_group': 8,
                  'n_early_every': 4, 'n_early_size': 2,
                  'WN_config': {'n_layers': 8, 'kernel_size': 3,
                                'n_channels': 512}}
        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v
            elif k in config['WN_config'].keys():
                config['WN_config'][k] = v

    m = waveglow.WaveGlow(**config)

    if fp16:
        m = batchnorm_to_float(m.half())
        for mat in m.convinv:
            mat.float()

    if pretrained:
        m.load_state_dict(state_dict)

    return m

def nvidia_ssd_processing_utils():
    import numpy as np
    import skimage
    from PyTorch.Detection.SSD.src.utils import dboxes300_coco, Encoder

    class Processing:
        @staticmethod
        def load_image(image_path):
            """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
            img = skimage.img_as_float(skimage.io.imread(image_path))
            if len(img.shape) == 2:
                img = np.array([img, img, img]).swapaxes(0, 2)
            return img

        @staticmethod
        def rescale(img, input_height, input_width):
            """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
            aspect = img.shape[1] / float(img.shape[0])
            if (aspect > 1):
                # landscape orientation - wide image
                res = int(aspect * input_height)
                imgScaled = skimage.transform.resize(img, (input_width, res))
            if (aspect < 1):
                # portrait orientation - tall image
                res = int(input_width / aspect)
                imgScaled = skimage.transform.resize(img, (res, input_height))
            if (aspect == 1):
                imgScaled = skimage.transform.resize(img, (input_width, input_height))
            return imgScaled

        @staticmethod
        def crop_center(img, cropx, cropy):
            """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
            y, x, c = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]

        @staticmethod
        def normalize(img, mean=128, std=128):
            img = (img * 256 - mean) / std
            return img

        @staticmethod
        def prepare_tensor(inputs, fp16=False):
            NHWC = np.array(inputs)
            NCHW = np.swapaxes(np.swapaxes(NHWC, 1, 3), 2, 3)
            tensor = torch.from_numpy(NCHW)
            tensor = tensor.cuda()
            tensor = tensor.float()
            if fp16:
                tensor = tensor.half()
            return tensor

        @staticmethod
        def prepare_input(img_uri):
            img = Processing.load_image(img_uri)
            img = Processing.rescale(img, 300, 300)
            img = Processing.crop_center(img, 300, 300)
            img = Processing.normalize(img)
            return img

        @staticmethod
        def decode_results(predictions):
            dboxes = dboxes300_coco()
            encoder = Encoder(dboxes)
            ploc, plabel = [val.float() for val in predictions]
            results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)
            return [[pred.detach().cpu().numpy() for pred in detections] for detections in results]

        @staticmethod
        def pick_best(detections, threshold=0.3):
            bboxes, classes, confidences = detections
            best = np.argwhere(confidences > threshold)[:, 0]
            return [pred[best] for pred in detections]

        @staticmethod
        def get_coco_object_dictionary():
            import os
            file_with_coco_names = "category_names.txt"

            if not os.path.exists(file_with_coco_names):
                print("Downloading COCO annotations.")
                import urllib
                import zipfile
                import json
                import shutil
                urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "cocoanno.zip")
                with zipfile.ZipFile("cocoanno.zip", "r") as f:
                    f.extractall()
                print("Downloading finished.")
                with open("annotations/instances_val2017.json", 'r') as COCO:
                    js = json.loads(COCO.read())
                class_names = [category['name'] for category in js['categories']]
                open("category_names.txt", 'w').writelines([c+"\n" for c in class_names])
                os.remove("cocoanno.zip")
                shutil.rmtree("annotations")
            else:
                class_names = open("category_names.txt").readlines()
                class_names = [c.strip() for c in class_names]
            return class_names

    return Processing()

def nvidia_ssd(pretrained=True, **kwargs):
    """Constructs an SSD300 model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com

    Args:
        pretrained (bool, True): If True, returns a model pretrained on COCO dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
    """

    from PyTorch.Detection.SSD.src import model as ssd

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    m = ssd.SSD300()
    if fp16:
        m = m.half()

        def batchnorm_to_float(module):
            """Converts batch norm to FP32"""
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()
            for child in module.children():
                batchnorm_to_float(child)
            return module

        m = batchnorm_to_float(m)

    if pretrained:
        if fp16:
            checkpoint = 'https://developer.nvidia.com/joc-ssd-fp16-pyt-20190225'
        else:
            checkpoint = 'https://developer.nvidia.com/joc-ssd-fp32-pyt-20190225'
        ckpt_file = os.path.basename(checkpoint)
        if not os.path.exists(ckpt_file) or force_reload:
            sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
            urllib.request.urlretrieve(checkpoint, ckpt_file)
        ckpt = torch.load(ckpt_file)
        ckpt = ckpt['model']
        if checkpoint_from_distributed(ckpt):
            ckpt = unwrap_distributed(ckpt)
        m.load_state_dict(ckpt)
    return m
