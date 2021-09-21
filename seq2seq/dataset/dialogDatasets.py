import random
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import json
from tqdm import tqdm

class TranslateData():
	def __init__(self, pad=0):
		self.pad = pad

	def collate_fn(self, batch):
		src = list(map(lambda x: x['src'], batch))
		tgt = list(map(lambda x: x['tgt'], batch))
		src_len = list(map(lambda x: x['src_len'], batch))
		tgt_len = list(map(lambda x: x['tgt_len'], batch))
		src = torch.transpose(pad_sequence(src, padding_value=self.pad), 0, 1)
		tgt = torch.transpose(pad_sequence(tgt, padding_value=self.pad), 0, 1)
		src_len = torch.stack(src_len)
		tgt_len = torch.stack(tgt_len)
		return {'src': src, 'tgt': tgt, 'src_len': src_len, 'tgt_len': tgt_len}

	def translate_data(self, subs, obj):
		import re
		import unicodedata
		def unicodeToAscii(s):
			return ''.join(
				c for c in unicodedata.normalize('NFD', s)
				if unicodedata.category(c) != 'Mn'
			)

		def normalizeString(s):
			s = unicodeToAscii(s.lower().strip())
			s = re.sub(r"([.!?])", r" \1", s)
			s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
			return s

		src, tgt = subs
		src = src.split(' ')
		tgt = tgt.split(' ')
		tgt = [obj.tgt_vocab.sos_token] + tgt + [obj.tgt_vocab.eos_token]
		if len(src) > obj.max_src_length or len(tgt) > obj.max_tgt_length:
			return None
		src_length, tgt_length = len(src), len(tgt)
		# src.extend([obj.src_vocab.pad_token] * (obj.max_src_length - src_length))
		# tgt.extend([obj.tgt_vocab.pad_token] * (obj.max_tgt_length - tgt_length))
		src_ids = [obj.src_vocab.word2idx[w] for w in src]
		tgt_ids = [obj.tgt_vocab.word2idx[w] for w in tgt]
		return {"src": torch.LongTensor(src_ids), 
				"tgt": torch.LongTensor(tgt_ids), 
				"src_len": torch.LongTensor([src_length]),
				"tgt_len": torch.LongTensor([tgt_length])}


class DialogDataset(Dataset):
	def __init__(self, src_data_fp, tgt_data_fp, transform_fuc, src_vocab, tgt_vocab, max_src_length, max_tgt_length):
		self.datasets = []
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.max_src_length = max_src_length
		self.max_tgt_length = max_tgt_length
		
		loaded = 0
		data_monitor = 0
		# cnt = 0
		with open(src_data_fp, 'r') as f_src, open(tgt_data_fp, 'r') as f_tgt:
			for line_src, line_tgt in tqdm(zip(f_src, f_tgt), desc="Load Data: "):
				src = line_src.strip()
				tgt = line_tgt.strip()
				subs = [src, tgt]
				loaded += 1
				if not data_monitor: data_monitor = len(subs)
				else: assert data_monitor == len(subs)
				item = transform_fuc(subs, self)
				if item: self.datasets.append(item)

				# if loaded == 10000:
				#     break

		print(f"{loaded} paris loaded. {len(self.datasets)} are valid. Rate {1.0 * len(self.datasets)/loaded:.4f}")

	
	def __len__(self):
		return len(self.datasets)

	def __getitem__(self, idx):
		return self.datasets[idx]


class myDialogDataset(Dataset):
	def __init__(self, dialogdir, limit,  src_data_fp, tgt_data_fp, transform_fuc, src_vocab, tgt_vocab, max_src_length, max_tgt_length):
		self.datasets = []
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.max_src_length = max_src_length
		self.max_tgt_length = max_tgt_length
		
		loaded = 0
		data_monitor = 0
		# cnt = 0
		# with open(src_data_fp, 'r') as f_src, open(tgt_data_fp, 'r') as f_tgt:
		#     for line_src, line_tgt in tqdm(zip(f_src, f_tgt), desc="Load Data: "):
		#         src = line_src.strip()
		#         tgt = line_tgt.strip()
		#         subs = [src, tgt]
		#         loaded += 1
		#         if not data_monitor: data_monitor = len(subs)
		#         else: assert data_monitor == len(subs)
		#         item = transform_fuc(subs, self)
		#         if item: self.datasets.append(item)

		#         if loaded == 10000:
		#             break
		self.filenames  =  sorted(os.listdir(dialogdir))
		print("there are %d users" % len(self.filenames))
		user_id = 0
		f_valid = open("/home/zhengyi_ma/pcb/Seq2Seq/val.txt","w")
		for filename in tqdm(self.filenames):
			user_id += 1
			fhand = open(os.path.join(dialogdir,filename))
			p_r = []
			for line in fhand:
				p, p_uid, p_time, r, r_uid, r_time, _, phase = line.strip().split("\t")
				src = p.strip()
				tgt = r.strip()
				
				subs = [src, tgt]
				p_r.append(subs)
				

				if not data_monitor: data_monitor = len(subs)
				else: assert data_monitor == len(subs)
			
			if len(p_r) > 10:
				for subs in p_r[:-2]:
					item = transform_fuc(subs, self)
					if item: self.datasets.append(item)
					loaded += 1
			
				f_valid.write(json.dumps(p_r[-2],ensure_ascii=False)+"\n")
				f_valid.write(json.dumps(p_r[-1],ensure_ascii=False)+"\n")
			if user_id > limit:
				break

		f_valid.close()


				
				
			
			
			
			

 
		print(f"{loaded} paris loaded. {len(self.datasets)} are valid. Rate {1.0 * len(self.datasets)/loaded:.4f}")

	
	def __len__(self):
		return len(self.datasets)

	def __getitem__(self, idx):
		return self.datasets[idx]

class AsyncDialogDataset(Dataset):
	def __init__(self, data_fp, load_num, transform_fuc, src_vocab, tgt_vocab, max_src_length, max_tgt_length, shuffle=False):
		self.async_flag = True
		self.src_vocab = src_vocab
		self.tgt_vocab = tgt_vocab
		self.max_src_length = max_src_length
		self.max_tgt_length = max_tgt_length
		self.transform_fuc = transform_fuc
		
		data_length = 0 
		# get the count of all samples
		with open(data_fp, 'r') as f:
			for _ in tqdm(f, desc="Get Data Length: "): data_length += 1
		self.data_fp = data_fp
		self.data_length = data_length
		self.shuffle = shuffle
		self.load_num = load_num

		self.finput_obj = open(self.data_fp, 'r')
		self.load_samples()
		print("Initialization Completed.")

	def load_samples(self):
		self.samples = list()
		# put n samples into memory
		while len(self.samples) < self.load_num:
			line = self.finput_obj.readline()
			if line:
				data = self.transform_fuc(line.strip().split('\t'), self)
				if data: self.samples.append(data)
			else:
				self.finput_obj = open(self.data_fp, 'r')
				# EOS initial dataset
		self.current_sample_num = len(self.samples)
		self.index = list(range(self.current_sample_num))
		if self.shuffle:
			random.shuffle(self.samples)
 
	def __len__(self):
		return self.data_length
 
	def __getitem__(self, item):
		idx = self.index[0]
		data = self.samples[idx]
		self.index = self.index[1:]
		self.current_sample_num -= 1
 
		if self.current_sample_num <= 0:
			self.load_samples()
			# all the samples in the memory have been used, need to get the new samples
		return data
