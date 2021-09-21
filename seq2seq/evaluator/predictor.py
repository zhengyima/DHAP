import torch
from torch.autograd import Variable


class Predictor(object):

	def __init__(self, model, src_vocab, tgt_vocab, device):
		"""
		Predictor class to evaluate for a given model.
		Args:
			model (seq2seq.models): trained model. This can be loaded from a checkpoint
				using `seq2seq.util.checkpoint.load`
			src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
			tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
		"""
		self.model = model
		self.device = device
		self.model.eval()
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab

	def get_decoder_features(self, src_seq):
		# src_id_seq = torch.LongTensor([self.src_vocab[tok] for tok in src_seq]).view(1, -1).to(self.device)
		# self.p_test = []
		# self.p_len_test = []
		# self.r_test = []
		# self.r_len_test = []
		# self.history_r_test = []
		# self.history_r_pos_test = []
		# self.r_mask_test = []
		# self.r_mask_post_test = []
		# self.history_p_test = []
		# self.history_p_pos_test = []
		# self.p_test_pos  = []
		p, p_len, r, r_len, history_r, history_r_pos, r_mask, r_mask_post, history_p, history_p_pos, p_pos = src_seq
		p = torch.LongTensor(p).to(self.device).unsqueeze(0)
		p_len = torch.LongTensor(p_len).to(self.device).unsqueeze(0)
		r = torch.LongTensor(r).to(self.device).unsqueeze(0)
		r_len = torch.LongTensor(r_len).to(self.device).unsqueeze(0)
		history_r = torch.LongTensor(history_r).to(self.device).unsqueeze(0)
		history_r_pos = torch.LongTensor(history_r_pos).to(self.device).unsqueeze(0)
		r_mask = torch.LongTensor(r_mask).to(self.device).unsqueeze(0)
		r_mask_post = torch.LongTensor(r_mask_post).to(self.device).unsqueeze(0)
		history_p = torch.LongTensor(history_p).to(self.device).unsqueeze(0)
		history_p_pos = torch.LongTensor(history_p_pos).to(self.device).unsqueeze(0)
		p_pos = torch.LongTensor(p_pos).to(self.device).unsqueeze(0)
		r_history_extend_vocab = r_history_extend_vocab.to(device)
		r_history_mask = r_history_mask.to(device)
		r_extend_vocab  = r_extend_vocab.to(device)
		r_max_oov = torch.LongTensor([max([len(oov) for oov in oovs])])
		# print(p.size(),p_len.size(),history_r.size(),history_r_pos.size(),r_mask.size(),r_mask_post.size(),p_pos.size(),history_p,size(),history_p_pos_test.size())


	
		with torch.no_grad():
			# softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
			softmax_list, _, other = self.model(p, p_len, r_history_idx=history_r, r_history_idx_pos=history_r_pos,
								r_mask=r_mask,  r_mask_pos=r_mask_post, p_pos = p_pos,  p_history_idx=history_p, p_history_idx_pos=history_p_pos_test,
								r_history_extend_vocab=r_history_extend_vocab, r_history_mask=r_history_mask, r_max_oov=r_max_oov, r_extend_vocab=r_extend_vocab)

		return other

	def predict(self, src_seq):
		""" Make prediction given `src_seq` as input.

		Args:
			src_seq (list): list of tokens in source language

		Returns:
			tgt_seq (list): list of tokens in target language as predicted
			by the pre-trained model
		"""
		other = self.get_decoder_features(src_seq)

		length = other['length'][0]

		tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
		tgt_seq = [self.tgt_vocab[tok.tolist()] for tok in tgt_id_seq]
		return tgt_seq

	def predict_n(self, src_seq, n=1):
		""" Make 'n' predictions given `src_seq` as input.

		Args:
			src_seq (list): list of tokens in source language
			n (int): number of predicted seqs to return. If None,
					 it will return just one seq.

		Returns:
			tgt_seq (list): list of tokens in target language as predicted
							by the pre-trained model
		"""
		other = self.get_decoder_features(src_seq)

		result = []
		for x in range(0, n):
			length = other['topk_length'][0][x]
			tgt_id_seq = [other['topk_sequence'][di][0][x].data[0] for di in range(length)]
			tgt_seq = [self.tgt_vocab[tok.tolist()] for tok in tgt_id_seq]
			result.append(tgt_seq)

		return result



class PerPredictor(object):

	def __init__(self, model, src_vocab, tgt_vocab, device):
		"""
		Predictor class to evaluate for a given model.
		Args:
			model (seq2seq.models): trained model. This can be loaded from a checkpoint
				using `seq2seq.util.checkpoint.load`
			src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
			tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
		"""
		self.model = model
		self.device = device
		self.model.eval()
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab

	def get_decoder_features(self, src_seq, i):
		# print("deco len:",len(src_seq))
		# print()

		p, p_len, r, r_len, history_r, history_r_pos, r_mask, r_mask_post, history_p, history_p_pos, p_pos = src_seq
		p = torch.LongTensor(p[i]).to(self.device).unsqueeze(0)
		p_len = torch.LongTensor(p_len[i]).to(self.device).unsqueeze(0)
		r = torch.LongTensor(r[i]).to(self.device).unsqueeze(0)
		r_len = torch.LongTensor(r_len[i]).to(self.device).unsqueeze(0)
		history_r = torch.LongTensor(history_r[i]).to(self.device).unsqueeze(0)
		history_r_pos = torch.LongTensor(history_r_pos[i]).to(self.device).unsqueeze(0)
		r_mask = torch.LongTensor(r_mask[i]).to(self.device).unsqueeze(0)
		r_mask_post = torch.LongTensor(r_mask_post[i]).to(self.device).unsqueeze(0)
		history_p = torch.LongTensor(history_p[i]).to(self.device).unsqueeze(0)
		history_p_pos = torch.LongTensor(history_p_pos[i]).to(self.device).unsqueeze(0)
		p_pos = torch.LongTensor(p_pos[i]).to(self.device).unsqueeze(0)


	
		with torch.no_grad():
			# softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
			softmax_list, _, other = self.model(p, p_len, None, history_r, history_r_pos, r_mask,  r_mask_post,  p_pos,  history_p, history_p_pos, istest=True)

		return other

	def predict(self, src_seq, i=None):
		""" Make prediction given `src_seq` as input.

		Args:
			src_seq (list): list of tokens in source language

		Returns:
			tgt_seq (list): list of tokens in target language as predicted
			by the pre-trained model
		"""
		other = self.get_decoder_features(src_seq,i)

		length = other['length'][0]

		tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
		tgt_seq = [self.tgt_vocab[tok.tolist()] for tok in tgt_id_seq]
		return tgt_seq

	def predict_n(self, src_seq, n=1,i=None):
		""" Make 'n' predictions given `src_seq` as input.

		Args:
			src_seq (list): list of tokens in source language
			n (int): number of predicted seqs to return. If None,
					 it will return just one seq.

		Returns:
			tgt_seq (list): list of tokens in target language as predicted
							by the pre-trained model
		"""
		other = self.get_decoder_features(src_seq,i)

		result = []
		for x in range(0, n):
			length = other['topk_length'][0][x]
			tgt_id_seq = [other['topk_sequence'][di][0][x].data[0] for di in range(length)]
			tgt_seq = [self.tgt_vocab[tok.tolist()] for tok in tgt_id_seq]
			result.append(tgt_seq)

		return result