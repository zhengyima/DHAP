from __future__ import print_function, division

import torch

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
	""" Class to evaluate models with given datasets.

	Args:
		loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
		batch_size (int, optional): batch size for evaluator (default: 64)
	"""

	def __init__(self, loss=NLLLoss(), batch_size=64, device=None, tgt_idx2word=None, tgt_vocab=None):
		self.loss = loss
		self.batch_size = batch_size
		self.device = device
		self.tgt_idx2word = tgt_idx2word
		self.tgt_vocab = tgt_vocab

	def evaluate(self, model, data, src_idx2word=None, tgt_idx2word=None):
		""" Evaluate a model on given dataset and return performance.

		Args:
			model (seq2seq.models): model to evaluate
			data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

		Returns:
			loss (float): loss of the given model on the given dataset
		"""
		model.eval()

		loss = self.loss
		device = self.device

		loss.reset()
		match = 0
		total = 0

		# tgt_vocab = data.dataset.tgt_vocab
		# pad = tgt_vocab.word2idx[tgt_vocab.pad_token]
		pad = 1

		with torch.no_grad():
			# for batch in data:
			allresult = []
			for p_idx, p_idx_len, r_idx, r_idx_len, r_history_idx, r_history_idx_pos, r_mask, r_mask_pos, p_pos, p_history_idx, \
			p_history_idx_pos in data:
				# src_variables = batch['src'].to(device)
				# tgt_variables = batch['tgt'].to(device)
				# src_lens = batch['src_len'].view(-1).to(device)
				# tgt_lens = batch['tgt_len'].view(-1).to(device)

				src_variables = p_idx.to(device)
				tgt_variables = r_idx.to(device)
				src_lens = p_idx_len.view(-1).to(device)
				tgt_lens = r_idx_len.view(-1).to(device)
				r_history_idx = r_history_idx.to(device)
				r_history_idx_pos = r_history_idx_pos.to(device)
				r_mask = r_mask.to(device)
				r_mask_pos = r_mask_pos.to(device)
				p_history_idx=  p_history_idx.to(device)
				p_history_idx_pos= p_history_idx_pos.to(device)
				p_pos = p_pos.to(device)
				# r_history_extend_vocab = r_history_extend_vocab.to(device)
				# r_history_mask = r_history_mask.to(device)
				# r_extend_vocab  = r_extend_vocab.to(device)
				# r_max_oov = torch.LongTensor([max([len(oov) for oov in oovs])])




				# decoder_outputs, decoder_hidden, other = model(src_variables, src_lens.tolist(), tgt_variables)
				decoder_outputs, decoder_hidden, other = model(src_variables, src_lens.tolist(), tgt_variables,
				r_history_idx, r_history_idx_pos,r_mask,  r_mask_pos, p_pos, p_history_idx, p_history_idx_pos, istest=False)

				# print(decoder_outputs)

				# Evaluation
				seqlist = other['sequence']
				# print("seqlist",seqlist)
				seq0_idxs = []
				batch_result = {}
				# weights = torch.ones(len(self.tgt_idx2word)+r_max_oov[0])
				# weights[self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]] = 0
				# weights = weights.to(device)
				# loss.criterion = torch.nn.NLLLoss(weight=weights, reduction=loss.reduction).to(device)
				for step, step_output in enumerate(decoder_outputs):
					seq0_idxs.append(seqlist[step][0])
					target = tgt_variables[:, step + 1]
					# new_target = r_extend_vocab[:, step+1]
					# loss.eval_batch(step_output.view(tgt_variables.size(0), -1), target)



					# print("eval.step_output.size():", step_output.size())
					# print("")
					# loss.eval_batch(step_output.view(r_extend_vocab.size(0), -1), target)
				
					loss.eval_batch(step_output.view(tgt_variables.size(0), -1), target)

					non_padding = target.ne(pad)
					correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
					match += correct
					total += non_padding.sum().item()
				
				batch_result['post'] = p_idx[0] # sl
				# print("batch_result", p_idx[0])
				batch_result['answer'] = r_idx[0]
				batch_result['result'] = seq0_idxs
				allresult.append(batch_result)

				# ("post:",p_idx,"answer",r_idx,"result:",seq0_idxs)
			#bs sl
				

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

		return loss.get_loss(), accuracy, allresult
