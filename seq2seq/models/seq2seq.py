import torch.nn as nn
import torch.nn.functional as F

import torch
import pickle
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from .Layers import EncoderLayer, DecoderLayer
from torch.autograd import Variable
from . import EncoderRNN, DecoderRNN

from .EncoderTransformer import Encoder_high

class Seq2seq(nn.Module):
		""" Standard sequence-to-sequence architecture with configurable encoder
		and decoder.

		Args:
				encoder (EncoderRNN): object of EncoderRNN
				decoder (DecoderRNN): object of DecoderRNN
				decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

		Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
				- **input_variable** (list, option): list of sequences, whose length is the batch size and within which
					each sequence is a list of token IDs. This information is forwarded to the encoder.
				- **input_lengths** (list of int, optional): A list that contains the lengths of sequences
						in the mini-batch, it must be provided when using variable length RNN (default: `None`)
				- **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
					each sequence is a list of token IDs. This information is forwarded to the decoder.
				- **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
					is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
					teacher forcing would be used (default is 0)

		Outputs: decoder_outputs, decoder_hidden, ret_dict
				- **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
					outputs of the decoder.
				- **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
					state of the decoder.
				- **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
					representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
					predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
					sequences, where each list is of attention weights }.

		"""

		# def __init__(self, encoder, decoder, encoder_r_history, decode_function=F.log_softmax):
		#     super(Seq2seq, self).__init__()
		#     # self.encoder = encoder
		#     # self.decoder = decoder


		#     self.decode_function = decode_function
		#     # self.encoder_r_history = 
		def __init__(self, en_vocab_size, en_max_len, en_embedding_size,en_rnn_cell,
						 en_n_layers,
						 en_hidden_size,
						 en_bidirectional, 
						 en_variable_lengths,de_vocab_size, de_max_len,
						 de_embedding_size,
						 de_rnn_cell,
						 de_n_layers,
						 de_hidden_size,
						 de_bidirectional,
						 de_dropout_p,
						 de_use_attention, 
						 de_use_knowledge,
						 de_eos_id, 
						 de_sos_id,
						 load_pretrain,
						 embedding_weight=None,
						 decode_function=F.log_softmax,
						 device=None,):

						super(Seq2seq, self).__init__()

						# print()
						 
						self.emb_p = nn.Embedding(en_vocab_size, en_embedding_size)
						self.emb_r = nn.Embedding(de_vocab_size, de_embedding_size)
						
						
						# if embedding_weight != None:
						if load_pretrain:
							embedding_weight = torch.tensor(embedding_weight, dtype=torch.float32)
							print(embedding_weight.size(),en_vocab_size,de_vocab_size,en_embedding_size)
							self.emb_p.weight = torch.nn.Parameter(embedding_weight)
							self.emb_r.weight = torch.nn.Parameter(embedding_weight)
						
							

						if de_use_knowledge:
							self.encoder_r_history = Encoder_high(
											len_max_seq=15*20+1,
											d_word_vec=en_embedding_size, d_model=en_embedding_size, d_inner=256,
											n_layers=1, n_head=6, d_k=50, d_v=50,
											dropout=0.1)
							self.encoder_p_history = Encoder_high(len_max_seq=20,d_word_vec=en_embedding_size, d_model=en_embedding_size, d_inner=256,
										n_layers=1, n_head=6, d_k=50, d_v=50,
										dropout=0.1)
						else:
							self.encoder_r_history = None
							self.encoder_p_history = None


						self.encoder  = EncoderRNN(en_vocab_size,
						 en_max_len,
						 embedding_size=en_embedding_size,
						 rnn_cell=en_rnn_cell,
						 n_layers=en_n_layers,
						 hidden_size=en_hidden_size,
						 bidirectional=en_bidirectional, 
						 variable_lengths=en_variable_lengths,
						 emb_p=self.emb_p,
						 emb_r=self.emb_r,
						 encoder_r_history=self.encoder_r_history,
						 encoder_p_history=self.encoder_p_history,
						 use_knowledge=de_use_knowledge)

						self.decoder = DecoderRNN(de_vocab_size,
						 de_max_len,
						 embedding_size=de_embedding_size,
						 rnn_cell=de_rnn_cell,
						 n_layers=de_n_layers,
						 hidden_size=de_hidden_size,
						 bidirectional=de_bidirectional,
						 dropout_p=de_dropout_p,
						 use_attention=de_use_attention, 
						 use_knowledge=de_use_knowledge,
						 eos_id=de_eos_id, 
						 sos_id=de_sos_id,
						 emb_r=self.emb_r,
						 encoder_r_history=self.encoder_r_history,
						 device=device)

						self.decode_function = decode_function
						self.device = device



		def flatten_parameters(self):
				self.encoder.rnn.flatten_parameters()
				self.decoder.rnn.flatten_parameters()

		def forward(self, input_variable, input_lengths=None, target_variable=None, r_history_idx=None, r_history_idx_pos=None,
								r_mask=None,  r_mask_pos=None, p_pos = None,  p_history_idx=None, p_history_idx_pos=None, teacher_forcing_ratio=0,  istest=None
								):
				
				encoder_outputs, encoder_hidden, all_renc_knowledge, all_rencs, all_pencs, all_rencs_byword = self.encoder(input_variable, input_lengths, r_history_idx=r_history_idx,
															r_history_idx_pos=r_history_idx_pos,
															r_mask=r_mask,r_mask_pos=r_mask_pos,p_pos=p_pos, p_history_idx=p_history_idx, p_history_idx_pos=p_history_idx_pos)
				# print("seqseq.forward.input_variable",input_variable.size())
				# print(r_history_idx.size())
				# print(r_history_idx)
				# print(teacher_forcing_ratio)
				result = self.decoder(inputs=target_variable, # bs, response_max_len
															encoder_hidden=encoder_hidden,
															encoder_outputs=encoder_outputs,
															function=self.decode_function,
															teacher_forcing_ratio=teacher_forcing_ratio,
															r_history_idx=r_history_idx,
															r_history_idx_pos=r_history_idx_pos,
															r_mask=r_mask,r_mask_pos=r_mask_pos, all_renc_knowledge=all_renc_knowledge,
															all_rencs=all_rencs, all_pencs=all_pencs, all_rencs_byword=all_rencs_byword, istest=istest)
				return result
