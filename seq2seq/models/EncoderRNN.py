import torch.nn as nn
import torch 
from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
	r"""
	Applies a multi-layer RNN to an input sequence.

	Args:
		vocab_size (int): size of the vocabulary
		max_len (int): a maximum allowed length for the sequence to be processed
		hidden_size (int): the number of features in the hidden state `h`
		input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
		dropout_p (float, optional): dropout probability for the output sequence (default: 0)
		n_layers (int, optional): number of recurrent layers (default: 1)
		bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
		rnn_cell (str, optional): type of RNN cell (default: gru)
		variable_lengths (bool, optional): if use variable length RNN (default: False)
		embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
			the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
			with the tensor if provided (default: None).
		update_embedding (bool, optional): If the embedding should be updated during training (default: False).

	Inputs: inputs, input_lengths
		- **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
		- **input_lengths** (list of int, optional): list that contains the lengths of sequences
			in the mini-batch, it must be provided when using variable length RNN (default: `None`)

	Outputs: output, hidden
		- **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
		- **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

	Examples::

		 >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
		 >>> output, hidden = encoder(input)

	"""

	def __init__(self, vocab_size, max_len, hidden_size, embedding_size,emb_p,emb_r,use_knowledge, encoder_r_history=None,encoder_p_history=None,
				 input_dropout_p=0, dropout_p=0,
				 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
				 embedding=None, update_embedding=True):
		super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size, embedding_size,
				input_dropout_p, dropout_p, n_layers, rnn_cell)
		self.variable_lengths = variable_lengths
		self.embedding = emb_p
		self.embedding_r = emb_r
		if embedding is not None:
			self.embedding.weight = nn.Parameter(embedding)
		self.embedding.weight.requires_grad = update_embedding
		self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
								 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
		self.encoder_r_history = encoder_r_history
		self.encoder_p_history = encoder_p_history
		self.use_knowledge = use_knowledge

	def forward(self, input_var, input_lengths=None, r_history_idx=None,
							  r_history_idx_pos=None,
							  r_mask=None,r_mask_pos=None,p_pos=None,p_history_idx=None,p_history_idx_pos=None):
		"""
		Applies a multi-layer RNN to an input sequence.

		Args:
			input_var (batch, seq_len): tensor containing the features of the input sequence.
			input_lengths (list of int, optional): A list that contains the lengths of sequences
			  in the mini-batch

		Returns: output, hidden
			- **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
			- **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
		"""
		all_renc_knowledge = None
		all_renc_byhistory = None
		all_penc = None
		all_renc = None
		arr_renc_byword = None
		if self.use_knowledge:
			r_history_embs = self.embedding_r(r_history_idx) # bs, history_max_len, response_max_len, emb_size
			r_history_embs = r_history_embs.view(-1, 15*20, self.embedding_size) # bs, history_max_len * response_max_len, emb_size
			r_history_idx_pos = r_history_idx_pos.view(-1, 15*20) # bs, history_max_len * response_max_len
			r_mask_embs = self.embedding_r(r_mask) # bs, 1, emb_size
			r_history_embs_withmask = torch.cat([r_history_embs, r_mask_embs], 1) # bs, history_max_len * response_max_len + 1, emb_size
			r_history_pos_withmask = torch.cat([r_history_idx_pos, r_mask_pos],1) # bs, history_max_len * response_max_len + 1
			all_renc, *_ = self.encoder_r_history(r_history_embs_withmask, r_history_pos_withmask, needpos=True) # bs, history_max_len * response_max_len + 1, d_model
			arr_renc_byword = all_renc
			all_renc_expand_nomask = all_renc[:,:-1,:].view(-1, 15, 20, self.embedding_size) # bs, history_max, response_max, emb_size
			all_renc_byhistory = torch.sum(all_renc_expand_nomask,2) # bs, history_max_len, d_Model
			all_renc_knowledge = all_renc[:,-1,:] # bs, d_model
			# p_history_idx: bs, history_max_len, post_max_len
			# p_history_idx_pos: bs, history_max_len, post_max_len
			p_history_embs = self.embedding(p_history_idx) #bs, history_max_len, post_max_len, emb_size
			p_history_embs  = p_history_embs.view(-1, 20, self.embedding_size) # bs*his, post_max_len, emb_size
			p_history_idx_pos = p_history_idx_pos.view(-1, 20) # bs*his, post_max
			all_penc, *_ = self.encoder_p_history(p_history_embs, p_history_idx_pos, needpos=True) # bs*his, post_max_len, emb_size
			all_penc = torch.sum(all_penc , 1)# bs*his, emb_size
			all_penc = all_penc.view(-1, 15, self.embedding_size) # bs, history_max_len, emb_size
		embedded = self.embedding(input_var)
		embedded = self.input_dropout(embedded)
		if self.variable_lengths:
			embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
		# embedded: bs, sl, emb_size
		output, hidden = self.rnn(embedded) 
		if self.variable_lengths:
			output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		
		return output, hidden, all_renc_knowledge, all_renc_byhistory, all_penc, arr_renc_byword