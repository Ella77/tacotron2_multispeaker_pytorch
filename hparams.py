import torch

class Hyperparameters():

	n_mel_channels = 80

	# Encoder
	symbols_embedding_size = 512
	style_embedding_size = 512
	speaker_embedding_size = 16

	# GST
	gst_n_tokens = 10
	gst_n_heads = 16

	# reference encoder
	ref_enc_filters = [32, 32, 64, 64, 128, 128]
	ref_enc_kernel_size = 3
	ref_enc_stride = 2
	ref_enc_pad = 1
	ref_enc_gru_size = style_embedding_size // 2
