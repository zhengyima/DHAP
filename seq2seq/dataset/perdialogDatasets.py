import random
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import json
from tqdm import tqdm
import numpy as np
import operator

class perDataset:

	# def __init__(self,src_vocab,tgt_vocab, data_path='/home/zhengyi_ma/pcb/Data/',
	# 			dialoglogfile='PChatbot_byuser_filter', word2vec_path='/home/zhengyi_ma/pcb/Data/PChatbot.word2vec.200d.txt',word2vec_dim=200,
	# 			limitation=300,max_history_len=15, batch_size=64, num_epoch=10 ,max_dec_steps=20,max_post_length=20,max_response_length=20):	
	def __init__(self,src_vocab,tgt_vocab, data_path, word2vec_dim=200,
				limitation=100,max_history_len=15, batch_size=64, num_epoch=10 ,max_dec_steps=20,max_post_length=20,max_response_length=20):	# reddit 40w
		# self.data_path = data_path
		# self.in_path = os.path.join(data_path, dialoglogfile)
		self.data_path = data_path
		self.in_path = data_path
		print("start loading log file list...")
		self.filenames  =  sorted(os.listdir(self.in_path))
		print("loading log file list complete, %d users" % (len(self.filenames)))
		# print(self.filenames[10])
		# print(self.filenames[100])

		self.word2vec_dim = word2vec_dim
		# self.word2vec_path = word2vec_path、
		self.limitation = limitation
		self.max_history_len = max_history_len
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.max_dec_steps = max_dec_steps
		self.max_post_length = max_post_length
		self.max_response_length = max_response_length
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		
		
		# # load word embedding
		# self.word2emb_dict = {} # word -> nparray
		# self.word2id = {"[PAD]":1,"[UNK]":0,"[START]":2,"[END]":3}
		# self.words = ["[UNK]","[PAD]","[START]","[END]"]
		# print("start loading word embeddings...")
		# with open(self.word2vec_path,"r") as f_wordemb:
		# 	cnt = 0
		# 	for line in f_wordemb:
		# 		cnt += 1
		# 		if cnt == 1:
		# 			self.vocab_size = int(line.split()[0]) + 4
		# 			# add [PAD] [UNK] [START] [END]
		# 			self.W_init = np.random.uniform(-0.25, 0.25, ((self.vocab_size), self.word2vec_dim)) # 直接用来初始化embedding矩阵
		# 			continue
		# 		else:
		# 			line_list = line.split()
		# 			word = line_list[0]
		# 			vector = np.fromstring(" ".join(line_list[1:]), dtype=float, sep=' ')
		# 			self.word2emb_dict[word] = vector
		# 			# cnt = 2对应第一个词 W_init中下标=4
		# 			self.W_init[cnt + 2] = vector
		# 			self.word2id[word] = cnt + 2
		# 			self.words.append(word)
		# print("loading word embeddings complete")
		
	
	def init_dataset(self):

		self.p_train = [] # current post
		self.p_len_train = [] # current post length
		self.r_train = []  # current response 
		self.r_len_train = []  # current response length
		self.history_r_train = []
		self.history_r_pos_train = []
		self.r_mask_train = []
		self.r_mask_post_train = []
		self.history_p_train = []
		self.history_p_pos_train = []
		self.p_train_pos = []
		# self.r_history_extend_vocab_train = []
		# self.oov_train = []
		# self.r_history_mask_train = []
		# self.r_extend_vocab_train = []

		self.p_dev = [] # current post
		self.p_len_dev = [] # current post length
		self.r_dev = []  # current response 
		self.r_len_dev = []  # current response length
		self.history_r_dev = []
		self.history_r_pos_dev = []
		self.r_mask_dev = []
		self.r_mask_post_dev = []
		self.history_p_dev = []
		self.history_p_pos_dev = []
		self.p_dev_pos = []
		# self.r_history_extend_vocab_dev = []
		# self.oov_dev = []
		# self.r_history_mask_dev = []
		# self.r_extend_vocab_dev = []



		self.p_test = []
		self.p_len_test = []
		self.r_test = []
		self.r_len_test = []
		self.history_r_test = []
		self.history_r_pos_test = []
		self.r_mask_test = []
		self.r_mask_post_test = []
		self.history_p_test = []
		self.history_p_pos_test = []
		self.p_test_pos  = []
		# self.r_history_extend_vocab_test = []
		# self.oov_test = []
		# self.r_history_mask_test = []
		# self.r_extend_vocab_test = []


	
	def trans_sentence_to_idx(self, sent, mode): 
		# 将句子转成idx_list
		idx_p = []
		# for w in sent.split(" "):
		# 	if w in self.word2id:
		# 		idx_p.append(self.word2id[w])
		# 	else:
		# 		idx_p.append(self.word2id['[UNK]'])
		sent_words = sent.split(" ")
		new_sent_words = []
		for w in sent_words:
			if len(w.strip()) > 0:
				new_sent_words.append(w)
		sent_words = new_sent_words
				

		ids = []
		oovs = []
		idx_extend_vocab = []
		if mode == 'p':
			idx_p = [ self.src_vocab.word2idx[w] for w in sent_words]
			max_length = self.max_post_length
			idx_p = idx_p[0:max_length]
			real_len = len(idx_p)
			idx_p.extend([self.src_vocab.word2idx[self.src_vocab.pad_token]] * (max_length - len(idx_p)))

			init_pos = np.zeros((self.max_post_length))
			for i in range(real_len):
				init_pos[i] = i+1
			

		elif mode == 'r':
			sent_words = [self.tgt_vocab.sos_token] + sent_words + [self.tgt_vocab.eos_token]
			idx_p = [ self.tgt_vocab.word2idx[w] for w in sent_words]
			max_length = self.max_response_length
			idx_p = idx_p[0:max_length]
			real_len = len(idx_p)

			idx_p.extend([self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]] * (max_length - len(idx_p)))
			init_pos = np.zeros((self.max_response_length))
			for i in range(real_len):
				init_pos[i] = i+1
			
			
			# unk_id = self.tgt_vocab.word2idx[self.tgt_vocab.unk_token]
			# sent_words_extend = [self.tgt_vocab.sos_token] + sent_words + [self.tgt_vocab.eos_token]
			# # idx_p_extend = [ self.tgt_vocab.word2idx[w] for w in sent_words]
			# for w in sent_words:
			# 	wi = self.tgt_vocab.word2idx[w]
			# 	if wi == unk_id:
			# 		if w not in oovs:
			# 			oovs.append(w)
			# 		oov_num = oovs.index(w)
			# 		ids.append(self.tgt_vocab.vocab_size + oov_num)
			# 	else:
			# 		ids.append(wi)
			# ids = ids[0:max_length]
			# ids.extend([self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]] * (max_length - len(ids)))
			
					
			
			
			

		return idx_p, real_len, init_pos, ids, oovs
	
	def can_as_data(self, r_time, last_r_time):
		if int(r_time) - int(last_r_time) > 1 * 60 * 10:
			return True
		else:
			return False

	# def prepare_single_data(self, p, r, label="train"):
	# 	# print([p],[r])
		

	# 	idx_p, idx_p_len = self.trans_sentence_to_idx(p,'p')
	# 	idx_r, idx_r_len = self.trans_sentence_to_idx(r,'r')


	# 	if label == "train":
	# 		self.p_train.append(idx_p)
	# 		self.p_len_train.append(idx_p_len)
	# 		self.r_train.append(idx_r)
	# 		self.r_len_train.append(idx_r_len)

	# 	elif label == "test":
	# 		self.p_test.append(idx_p)
	# 		self.p_len_test.append(idx_p_len)
	# 		self.r_test.append(idx_r)
	# 		self.r_len_test.append(idx_r_len)
		

	# 	# print([idx_p],[idx_r])
	
	# def prepare_single_data_new(self, idx_p, idx_p_len, idx_r, idx_r_len, histroy_r_list, history_p_list):
	def prepare_single_data_new(self, data, label):

		idx_p, idx_p_len, idx_r, idx_r_len, histroy_r_list, history_p_list, history_r_src_list, p_pos, r_pos, r_src = data
		# print("history_r_list", histroy_r_list)
		# print("histroy_r_src_list", history_r_src_list)

		init_history_pids = np.ones((self.max_history_len,self.max_post_length))
		init_history_rids = np.ones((self.max_history_len,self.max_response_length))
		init_history_rids_extend_vocab = np.ones((self.max_history_len, self.max_response_length))
		init_oovs = []


		init_history_pos = np.zeros((self.max_history_len, self.max_response_length))
		init_history_pos_post = np.zeros((self.max_history_len, self.max_post_length))


		# init_history_pos[-1] = self.max_history_len+1
		history_r_his = histroy_r_list[-self.max_history_len:]
		history_p_his = history_p_list[-self.max_history_len:]
		history_r_src_his = history_r_src_list[-self.max_history_len:]
		

		# ids = []
		# oovs = []
		# unk_id = self.tgt_vocab.word2idx[self.tgt_vocab.unk_token]
		# for sent in history_r_src_his:

		# 	sent_ids = []

		# 	sent_words = sent.split(" ")
		# 	new_sent_words = []
		# 	for w in sent_words:
		# 		if len(w.strip()) > 0:
		# 			new_sent_words.append(w)
		# 	sent_words = new_sent_words
		# 	sent_words = [self.tgt_vocab.sos_token] + sent_words + [self.tgt_vocab.eos_token]
		# 	sent_words = sent_words[0:self.max_response_length]
			

		# 	for w in sent_words:
		# 		i = self.tgt_vocab.word2idx[w]
		# 		if i == unk_id:
		# 			if w not in oovs:
		# 				oovs.append(w)
		# 			oov_num = oovs.index(w)
		# 			sent_ids.append(self.tgt_vocab.vocab_size + oov_num)
		# 		else:
		# 			sent_ids.append(i)

		# 	sent_ids = sent_ids[0:self.max_response_length]


			
		# 	sent_ids.extend([self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]] * (self.max_response_length - len(sent_ids)))
		# 	ids.extend(sent_ids)
		
	
		# for i in range(self.max_history_len-len(history_r_src_his)):
		# 	ids.extend([1] * self.max_response_length)

		# print(ids)
		# assert max(ids) == self.tgt_vocab.vocab_size + len(oovs)
		
		# init_ids_mask = []
		# for i in ids:
		# 	if i in [self.tgt_vocab.word2idx[self.tgt_vocab.pad_token], self.tgt_vocab.word2idx[self.tgt_vocab.sos_token], 
		# 		self.tgt_vocab.word2idx[self.tgt_vocab.eos_token]]:
		# 		init_ids_mask.append(0)
		# 	else:
		# 		init_ids_mask.append(1)
		# print(init_ids_mask)
		# ids.extend( [[1] * self.max_response_length] * (self.max_history_len-len(history_r_src_his)) )
		
	
		


		for i in range(len(history_r_his)):
			init_history_rids[i] = history_r_his[i]
			init_history_pids[i] = history_p_his[i]

			j_pad_idx = self.max_response_length # 记录pad开始的index
			for j in range(self.max_response_length):
				if history_r_his[i][j] == 1:
					j_pad_idx = j
					break

			j_pad_idx_p = self.max_post_length
			for j in range(self.max_post_length):
				if history_p_his[i][j] == 1:
					j_pad_idx_p = j
					break

			for j in range(j_pad_idx):
				init_history_pos[i][j] = j + 1
				init_history_pos[i][-1] = self.max_response_length

			for j in range(j_pad_idx_p):
				init_history_pos_post[i][j] = j + 1

		# print("---------------------------------------")

		# print("ids",ids,len(ids))
		# print(history_r_src_list)
		# print("oovs",oovs)
		
		# print("init_his_r", init_history_rids.tolist())
		# print("mask",init_ids_mask)
		# print("---------------------------------------")
		# print(len())
		

		# sent_words = r_src.split(" ")
		# new_sent_words = []
		# for w in sent_words:
		# 	if len(w.strip()) > 0:
		# 		new_sent_words.append(w)
		# sent_words = new_sent_words
		# sent_words = [self.tgt_vocab.sos_token] + sent_words + [self.tgt_vocab.eos_token]
		# sent_ids = []
		# for w in sent_words:
		# 	i = self.tgt_vocab.word2idx[w]
		# 	if i == unk_id:
		# 		if w in oovs:
		# 			oov_num = oovs.index(w)
		# 			i = self.tgt_vocab.vocab_size + oov_num
		# 	sent_ids.append(i)
		# sent_ids = sent_ids[0:self.max_response_length]
		# sent_ids.extend([self.tgt_vocab.word2idx[self.tgt_vocab.pad_token]] * (self.max_response_length - len(sent_ids)))
		
		# if not operator.eq(sent_ids, idx_r):
		# 	print("sent_ids",sent_ids)
		# 	print("idx_r", idx_r)


		# for i in range(self.)
		# assert len(ids) == len(init_ids_mask) == self.max_history_len * self.max_response_length
		# assert len(sent_ids) == len(idx_r)
		# assert max(ids) == self.tgt_vocab.vocab_size + len(oovs) - 1 or (len(oovs) == 0)
		# assert max(sent_ids) < self.tgt_vocab.vocab_size + len(oovs)
		# assert len(sent_ids) == len(idx_r)
		
		if label == 'train':
			self.p_train.append(idx_p)
			self.p_len_train.append(idx_p_len)
			self.r_train.append(idx_r)
			self.r_len_train.append(idx_r_len)
			self.history_r_train.append(init_history_rids)
			self.history_r_pos_train.append(init_history_pos)
			self.history_p_train.append(init_history_pids)
			self.history_p_pos_train.append(init_history_pos_post)
			self.r_mask_train.append([4])
			self.r_mask_post_train.append([1]) 
			self.p_train_pos.append(p_pos)
			# self.r_history_extend_vocab_train.append(ids)
			# self.r_history_mask_train.append(init_ids_mask)
			# self.oov_train.append(oovs)
			# self.r_extend_vocab_train.append(idx_r)
			# print("history_r_train",init_history_rids)
			# print("r_extend_vocab_train",ids)
			# print("oovs", oovs, len(oovs))
			# print("max history_r_train",max(ids))
			# print("-----------------------------------------")

			# self.r_extend_vocab_train.append(idx_r)

		elif label == "dev":
			self.p_dev.append(idx_p)
			self.p_len_dev.append(idx_p_len)
			self.r_dev.append(idx_r)
			self.r_len_dev.append(idx_r_len)
			self.history_r_dev.append(init_history_rids)
			self.history_r_pos_dev.append(init_history_pos)
			self.history_p_dev.append(init_history_pids)
			self.history_p_pos_dev.append(init_history_pos_post)
			self.r_mask_dev.append([4])
			self.r_mask_post_dev.append([1]) 
			self.p_dev_pos.append(p_pos)
			# self.r_history_extend_vocab_dev.append(ids)		
			# self.r_history_mask_dev.append(init_ids_mask)
			# self.oov_dev.append(oovs)	
			# self.r_extend_vocab_dev.append(idx_r)
			# self.r_extend_vocab_dev.append(idx_r)

		elif label == 'test':
			self.p_test.append(idx_p)
			self.p_len_test.append(idx_p_len)
			self.r_test.append(idx_r)
			self.r_len_test.append(idx_r_len)
			self.history_r_test.append(init_history_rids)
			self.history_r_pos_test.append(init_history_pos)
			self.history_p_test.append(init_history_pids)
			self.history_p_pos_test.append(init_history_pos_post)
			self.r_mask_test.append([4])
			self.r_mask_post_test.append([1])
			self.p_test_pos.append(p_pos)
			# self.r_history_mask_test.append(init_ids_mask)
			# self.oov_test.append(oovs)	
			# self.r_extend_vocab_test.append(sent_ids)
		else:
			assert False 			
		                                 




	def prepare_dataset(self): 
		# 往init_dataset里准备的容器里灌数据
		if not hasattr(self, "p_train"):
			self.init_dataset()
		
		user_id = 0
		print("There are %d users in the log directory" % (len(self.filenames)))
		label = "train"
		ne = 0
		n = 0
		for filename in tqdm(self.filenames):

			user_id += 1
			# if user_id == len(self.filenames): # 最后一个用户用来测试
			# 	label = "test"

			last_r_time = 0 # 上一条response的时间
			fhand = open(os.path.join(self.in_path,filename))
			p_r_list = []
			history_r_list = []
			history_p_list = []
			history_r_src_list = []


			# n += 1
			if user_id % 1000 == 0:
				print("user_id,ne:",user_id,ne)

			single_datas =  []
			for line in fhand:
				# print(len(line.strip().split("\t")))
				try:
					p, p_uid, p_time, r, r_uid, r_time, _, phase = line.strip().split("\t")
					p = p.strip().lower()
					r = r.strip().lower()
				except:
					ne += 1
					continue
				
				# if self.can_as_data(r_time, last_r_time): # 这条数据可以作为一条数据
					# p_r_list.append([p,r])
					# self.prepare_single_data(p, r, label) # 生成一条数据
				idx_p, idx_p_len, p_pos, _, _ = self.trans_sentence_to_idx(p,'p')
				idx_r, idx_r_len, r_pos, idx_r_extend_vocab, r_oovs = self.trans_sentence_to_idx(r,'r')
				
				# print(idx_r, idx_r_extend_vocab, r_oovs)
				
				single_datas.append([idx_p, idx_p_len, idx_r, idx_r_len, history_r_list[0:], history_p_list[0:], history_r_src_list[0:], 
				p_pos, r_pos, r])

				history_r_list.append(idx_r)
				history_p_list.append(idx_p)
				history_r_src_list.append(r)

				# self.prepare_single_data_new()

				
				last_r_time = r_time
			
			
			for d in single_datas[:-4]:
				self.prepare_single_data_new(d,'train')
			for d in single_datas[-4:-2]:
				self.prepare_single_data_new(d,'dev')
			for d in single_datas[-2:]:
				self.prepare_single_data_new(d,'test')			
			


			

			# for i in range(len(p_r_list)-2):
			# 	self.prepare_single_data(p_r_list[i][0],p_r_list[i][1],'train' )

			# if len(p_r_list) > 3:
			# 	self.prepare_single_data(p_r_list[-2][0],p_r_list[-2][1],'test' )
			# 	self.prepare_single_data(p_r_list[-1][0],p_r_list[-1][1],'test' )

			
			if user_id > self.limitation:
				break
		# print("n:",n,"ne:",ne)


class Dataset_train(torch.utils.data.Dataset):
	def __init__(
		self, p_idx, p_idx_len, r_idx, r_idx_len,r_history_idx,
			r_history_idx_pos, r_mask, r_mask_pos,p_pos,p_history_idx,p_history_idx_pos):
		self.p_idx = p_idx
		self.p_idx_len = p_idx_len
		self.r_idx = r_idx
		self.r_idx_len = r_idx_len
		self.r_history_idx = r_history_idx
		self.r_history_idx_pos = r_history_idx_pos
		self.p_history_idx = p_history_idx
		self.p_history_idx_pos = p_history_idx_pos
		self.r_mask = r_mask
		self.r_mask_pos = r_mask_pos
		self.p_pos = p_pos

		# self.r_history_extend_vocab = r_history_extend_vocab
		# self.oov = oov
		# self.r_history_mask= r_history_mask
		# self.r_extend_vocab = r_extend_vocab


	def __len__(self):
		return len(self.p_idx)

	def __getitem__(self, idx):
		p_idx = self.p_idx[idx]
		p_idx_len = self.p_idx_len[idx]
		r_idx = self.r_idx[idx]
		r_idx_len = self.r_idx_len[idx]
		r_history_idx = self.r_history_idx[idx]
		r_history_idx_pos = self.r_history_idx_pos[idx]
		p_history_idx = self.p_history_idx[idx]
		p_history_idx_pos = self.p_history_idx_pos[idx]
		r_mask = self.r_mask[idx]
		r_mask_pos = self.r_mask_pos[idx]
		p_pos = self.p_pos[idx]
		# r_history_extend_vocab = self.r_history_extend_vocab[idx]
		# oov = self.oov[idx]
		# r_history_mask = self.r_history_mask[idx]
		# r_extend_vocab = self.r_extend_vocab[idx]



	
		return p_idx, p_idx_len, r_idx, r_idx_len, r_history_idx, r_history_idx_pos, r_mask, r_mask_pos, p_pos,p_history_idx,p_history_idx_pos