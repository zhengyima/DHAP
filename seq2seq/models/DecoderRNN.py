import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention, Attention_Knowledge, NaiveAttention
from .baseRNN import BaseRNN
from .EncoderTransformer import Encoder_high

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
	""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
		Args:
			logits: logits distribution shape (vocabulary size)
			top_k >0: keep only top k tokens with highest probability (top-k filtering).
			top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
				Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
	"""
	assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
	top_k = min(top_k, logits.size(-1))  # Safety check
	if top_k > 0:
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p > 0.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold
		sorted_indices_to_remove = cumulative_probs > top_p
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits[indices_to_remove] = filter_value
	return logits

class DecoderRNN(BaseRNN):
	r"""
	Provides functionality for decoding in a seq2seq framework, with an option for attention.

	Args:
		vocab_size (int): size of the vocabulary
		max_len (int): a maximum allowed length for the sequence to be processed
		hidden_size (int): the number of features in the hidden state `h`
		sos_id (int): index of the start of sentence symbol
		eos_id (int): index of the end of sentence symbol
		n_layers (int, optional): number of recurrent layers (default: 1)
		rnn_cell (str, optional): type of RNN cell (default: gru)
		bidirectional (bool, optional): if the encoder is bidirectional (default False)
		input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
		dropout_p (float, optional): dropout probability for the output sequence (default: 0)
		use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

	Attributes:
		KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
		KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
		KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

	Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
		- **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
		  each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
		- **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
		  hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
		- **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
		  Used for attention mechanism (default is `None`).
		- **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
		  (default is `torch.nn.functional.log_softmax`).
		- **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
		  drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
		  teacher forcing would be used (default is 0).

	Outputs: decoder_outputs, decoder_hidden, ret_dict
		- **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
		  the outputs of the decoding function.
		- **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
		  state of the decoder.
		- **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
		  representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
		  predicted token IDs }.
	"""

	KEY_ATTN_SCORE = 'attention_score'
	KEY_LENGTH = 'length'
	KEY_SEQUENCE = 'sequence'

	def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
			sos_id, eos_id, emb_r, encoder_r_history, n_layers=1,   rnn_cell='gru', bidirectional=False,
			input_dropout_p=0, dropout_p=0, use_attention=False, use_knowledge=False, device=None):
		super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size, embedding_size,
				input_dropout_p, dropout_p, n_layers, rnn_cell)

		self.bidirectional_encoder = bidirectional
		self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
		self.output_size = vocab_size
		self.max_length = max_len
		self.use_attention = use_attention
		self.eos_id = eos_id
		self.sos_id = sos_id
		self.use_knowledge = use_knowledge
		self.device = device
		self.init_input = None
		self.embedding = emb_r

		if use_attention:
			if use_knowledge:
				self.attention = Attention_Knowledge(self.hidden_size, embedding_size)
				self.naiveattention = NaiveAttention()
			else:
				self.attention = Attention(self.hidden_size)

		self.out = nn.Linear(self.hidden_size, self.output_size)
		self.for_word_level_att_linear = nn.Linear(self.hidden_size, self.embedding_size)
		self.decoding_switcher_mlp = nn.Linear(2 * (self.hidden_size + embedding_size), 2)
		if use_knowledge:
			self.encoder_r_history = encoder_r_history

	def forward_step(self, input_var, hidden, encoder_outputs, function, knowledge, pr_memory, 
	all_rencs_byword,r_history_idx):
		all_rencs_byword = all_rencs_byword[:, :-1, :]
		batch_size = input_var.size(0)
		output_size = input_var.size(1)
		embedded = self.embedding(input_var)
		embedded = self.input_dropout(embedded)
		output, hidden = self.rnn(embedded, hidden)
		all_p_encodes, all_r_encodes = pr_memory
		# all_r_encodes : bs, history_max, word_vec
		# output: bs, decode_max_step, hidden_num
		decode_features = self.for_word_level_att_linear(output)
		# decode_features: bs, decode_max_step, word_vec
		copy_attn = self.naiveattention(decode_features, all_rencs_byword) #  bs, decode_max_step, history_max
		attn = None
		if self.use_attention:
			if self.use_knowledge:
				output, attn, combined = self.attention(output, encoder_outputs, knowledge, pr_memory)
			else:
				output, attn = self.attention(output, encoder_outputs)
		output_flatten = self.out(output.contiguous().view(-1, self.hidden_size)) # bs* sl, vocab_size
		output_flatten_softmax = F.softmax(output_flatten,dim=1) # bs*sl, vocab_size
		vocab_dists =  output_flatten_softmax.view(batch_size, output_size, -1) # bs, sl, vocab_size
		attn_dists = copy_attn # bs, sl, history_max
		_enc_batch_extend_vocab = r_history_idx.view(batch_size, -1) # bs, history_max
		vocab_dists = torch.transpose(vocab_dists, 0, 1) # sl, bs, vocab_size
		attn_dists = torch.transpose(attn_dists, 0, 1) # sl, bs, history_max

		combined = torch.transpose(combined, 0, 1) # sl, bs, dim
		decoding_probs = F.softmax(self.decoding_switcher_mlp(combined), dim=2) # sl, bs, 2
		vocab_dists = decoding_probs[:,:,0:1] * vocab_dists # sl, bs vocab_size
		attn_dists = decoding_probs[:,:,1:2] * attn_dists # sl, bs, history_max
		extended_vsize = vocab_dists.size(2)
		vocab_dists_extend = [ dist for dist in vocab_dists] # [(batch_size, extended_vsize) ,(batch_size, extended_vsize) ...]
		batch_nums = torch.arange(start=0, end=batch_size).to(self.device) # shape (batch_size) [0,1,2...63]
		batch_nums = torch.unsqueeze(batch_nums, 1) # shape (batch_size, 1) # [[0],[1],[2],....[63]]
		attn_len = _enc_batch_extend_vocab.size(1) # history_max
		batch_nums = batch_nums.repeat(1, attn_len) # shape (batch_size, attn_len)
		indices = torch.stack((batch_nums,_enc_batch_extend_vocab),dim=2)  # shape (batch_size, attn_len, 2)
		shape = [batch_size, extended_vsize]
		batch_nums_flateen = batch_nums.view(-1) # shape (batch_size * attn_len)
		_enc_batch_extend_vocab_flatten = _enc_batch_extend_vocab.view(-1)
		index_tuple = (batch_nums_flateen, _enc_batch_extend_vocab_flatten)
		# sl, bs, extend_vsize
		attn_dists_projected = [(torch.zeros([batch_size, extended_vsize])+1e-7).to(self.device).index_put_(index_tuple, copy_dist.contiguous().view(-1),True) for copy_dist in attn_dists]
		word_level_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extend, attn_dists_projected)]
		word_level_dists = torch.stack(word_level_dists)
		_output_softmax_extend_log = torch.log(word_level_dists + 1e-7)
		# copy_dist: bs, history_max
		res_outputs = torch.transpose(_output_softmax_extend_log,0,1)
		return res_outputs, hidden, attn

	def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
					function=F.log_softmax, teacher_forcing_ratio=0,r_history_idx=None,
							  r_history_idx_pos=None,
							  r_mask=None,r_mask_pos=None,all_renc_knowledge=None,all_rencs=None,all_pencs=None,all_rencs_byword=None,istest=None):
		ret_dict = dict()
		if self.use_attention:
			ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
		inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
															 function, teacher_forcing_ratio)
		decoder_hidden = self._init_state(encoder_hidden)
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		decoder_outputs = []
		sequence_symbols = []
		lengths = np.array([max_length] * batch_size)

		def decode(step, step_output, step_attn):
			decoder_outputs.append(step_output)
			if self.use_attention:
				ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
			symbols = decoder_outputs[-1].topk(1)[1]
			sequence_symbols.append(symbols)

			eos_batches = symbols.data.eq(self.eos_id)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
			return symbols

		def decode_test(step, step_output, step_attn):
			decoder_outputs.append(torch.unsqueeze(top_k_top_p_filtering(step_output[0], top_p=0.8), 0))
			if self.use_attention:
				ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
			symbols = torch.unsqueeze(torch.multinomial(F.softmax(decoder_outputs[-1][0],dim=-1), 1),0)
			sequence_symbols.append(symbols)
			eos_batches = symbols.data.eq(self.eos_id)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
			return symbols

		# Manual unrolling is used to support random teacher forcing.
		# If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
		if use_teacher_forcing:
			decoder_input = inputs[:, :-1]
			decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
																	 function=function,knowledge=all_renc_knowledge, pr_memory=[all_pencs, all_rencs],
																	 all_rencs_byword=all_rencs_byword,r_history_idx=r_history_idx)
			for di in range(decoder_output.size(1)):
				step_output = decoder_output[:, di, :]
				if attn is not None:
					step_attn = attn[:, di, :]
				else:
					step_attn = None
				decode(di, step_output, step_attn)
		else:
			decoder_input = inputs[:, 0].unsqueeze(1)
			for di in range(max_length):
				decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
																		 function=function, knowledge=all_renc_knowledge, 
																		 pr_memory=[all_pencs, all_rencs],all_rencs_byword=all_rencs_byword,
																		 r_history_idx=r_history_idx)
				step_output = decoder_output.squeeze(1)
				if istest:
					symbols = decode_test(di, step_output, step_attn)
				else:
					symbols = decode(di, step_output, step_attn)
				decoder_input = symbols

		ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
		ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

		return decoder_outputs, decoder_hidden, ret_dict

	def _init_state(self, encoder_hidden):
		""" Initialize the encoder hidden state. """
		if encoder_hidden is None:
			return None
		if isinstance(encoder_hidden, tuple):
			encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
		else:
			encoder_hidden = self._cat_directions(encoder_hidden)
		return encoder_hidden

	def _cat_directions(self, h):
		""" If the encoder is bidirectional, do the following transformation.
			(#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
		"""
		if self.bidirectional_encoder:
			h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
		return h

	def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
		if self.use_attention:
			if encoder_outputs is None:
				raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

		# inference batch size
		if inputs is None and encoder_hidden is None:
			batch_size = 1
		else:
			if inputs is not None:
				batch_size = inputs.size(0)
			else:
				if self.rnn_cell is nn.LSTM:
					batch_size = encoder_hidden[0].size(1)
				elif self.rnn_cell is nn.GRU:
					batch_size = encoder_hidden.size(1)

		# set default input and max decoding length
		if inputs is None:
			if teacher_forcing_ratio > 0:
				raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
			device = encoder_hidden.device if encoder_hidden is not None else torch.device('cpu')
			inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
			inputs = inputs.to(device)
			max_length = self.max_length
		else:
			max_length = inputs.size(1) - 1 # minus the start of sequence symbol

		return inputs, batch_size, max_length
