import os,logging,torch,json,pickle
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder,seq2seq
from seq2seq.models.seq2seq import Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import VocabField
from seq2seq.evaluator import Predictor
from seq2seq.evaluator.predictor import PerPredictor
from configParser import opt
from seq2seq.dataset.perdialogDatasets import *
from seq2seq.models.EncoderTransformer import Encoder_high
from tqdm import tqdm

if opt.random_seed is not None: torch.cuda.manual_seed_all(opt.random_seed)

multi_gpu = False
device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')

logger = logging.getLogger('train')

def get_last_checkpoint(model_dir):
	checkpoints_fp = os.path.join(model_dir, "checkpoints")
	try:
		with open(checkpoints_fp, 'r') as f:
			checkpoint = f.readline().strip()
	except:
		return None
	return checkpoint


def collate_fn_train(insts):
	''' Pad the instance to the max seq length in batch '''
	p_idx, p_idx_len, r_idx, r_idx_len, r_history_idx, r_history_idx_pos, r_mask, r_mask_pos, p_pos, p_history_idx, p_history_idx_pos = zip(*insts)

	p_idx = torch.LongTensor(p_idx)
	p_idx_len = torch.LongTensor(p_idx_len)
	r_idx = torch.LongTensor(r_idx)
	r_idx_len = torch.LongTensor(r_idx_len)
	r_history_idx = torch.LongTensor(r_history_idx)
	r_history_idx_pos = torch.LongTensor(r_history_idx_pos)
	r_mask = torch.LongTensor(r_mask)
	r_mask_pos = torch.LongTensor(r_mask_pos)
	p_pos = torch.LongTensor(p_pos)
	p_history_idx = torch.LongTensor(p_history_idx)
	p_history_idx_pos = torch.LongTensor(p_history_idx_pos)
	return p_idx, p_idx_len, r_idx, r_idx_len, r_history_idx, r_history_idx_pos, r_mask, r_mask_pos, p_pos, p_history_idx, p_history_idx_pos

if __name__ == "__main__":
	load_pretrain = True
	if load_pretrain:
		emb_file = opt.word2vec_path
		src_vocab_list, embs = VocabField.load_from_pretrained(emb_file)
		src_vocab_list = src_vocab_list[:40000]
		embs = embs[:40000]
		tgt_vocab_list = src_vocab_list
		src_vocab = VocabField(src_vocab_list, vocab_size=opt.src_vocab_size)
		tgt_vocab = VocabField(tgt_vocab_list, vocab_size=opt.tgt_vocab_size, 
								sos_token="<SOS>", eos_token="<EOS>")
		embedding_weight = torch.from_numpy(embs[:src_vocab.vocab_size])
	
	pad_id = tgt_vocab.word2idx[tgt_vocab.pad_token]
	# 0: UNK, 1: PAD, 2:START, 3:END
	# Prepare loss
	weight = torch.ones(len(tgt_vocab.vocab))
	loss = Perplexity(weight, pad_id)
	loss.to(device)

	print("en_vocab_size",len(src_vocab.vocab))
	print("en_vocab_size",len(tgt_vocab.vocab))
	seq2seq = Seq2seq(en_vocab_size=len(src_vocab.vocab), en_max_len=opt.max_src_length, en_embedding_size=opt.embedding_size,en_rnn_cell=opt.rnn_cell,
						en_n_layers=opt.n_hidden_layer,
						en_hidden_size=opt.hidden_size,
						en_bidirectional=opt.bidirectional, 
						en_variable_lengths=False,de_vocab_size=len(tgt_vocab.vocab), de_max_len=opt.max_tgt_length,
						de_embedding_size=opt.embedding_size,
						de_rnn_cell=opt.rnn_cell,
						de_n_layers=opt.n_hidden_layer,
						de_hidden_size=opt.hidden_size * 2 if opt.bidirectional else opt.hidden_size,
						de_bidirectional=opt.bidirectional,
						de_dropout_p=0.2,
						de_use_attention=opt.use_attn, 
						de_use_knowledge=opt.use_know,
						de_eos_id=tgt_vocab.word2idx[tgt_vocab.eos_token], 
						de_sos_id=tgt_vocab.word2idx[tgt_vocab.sos_token],
						device=device,
						embedding_weight=embedding_weight,
						load_pretrain=load_pretrain
					)
	
	seq2seq.to(device)

	if opt.resume and not opt.load_checkpoint:
		last_checkpoint = get_last_checkpoint(opt.model_dir)
		if last_checkpoint:
			opt.load_checkpoint = os.path.join(opt.model_dir, last_checkpoint)
			opt.skip_steps = int(last_checkpoint.strip('.pt').split('/')[-1])

	if opt.load_checkpoint:
		seq2seq.load_state_dict(torch.load(opt.load_checkpoint))
		opt.skip_steps = int(opt.load_checkpoint.strip('.pt').split('/')[-1])
	else:
		for param in seq2seq.parameters():
			param.data.uniform_(-opt.init_weight, opt.init_weight)
	
	if opt.beam_width > 1 and opt.phase == "infer":
		seq2seq.decoder = TopKDecoder(seq2seq.decoder, opt.beam_width, opt.use_know)

	if opt.phase == "train":
		# Prepare Train Data
		perdiadata = perDataset(src_vocab,tgt_vocab, data_path=opt.data_path, limitation=opt.user_limit)
		perdiadata.init_dataset()
		perdiadata.prepare_dataset()
		
		test_data = [perdiadata.p_test, perdiadata.p_len_test, perdiadata.r_test, perdiadata.r_len_test, perdiadata.history_r_test,
			perdiadata.history_r_pos_test, perdiadata.r_mask_test, perdiadata.r_mask_post_test, perdiadata.history_p_test, perdiadata.history_p_pos_test,
			perdiadata.p_test_pos		
		]
		print("%d train pairs loaded, %d dev pairs loaded, %d test pairs loaded" % (len(perdiadata.r_train), len(perdiadata.r_dev) ,len(perdiadata.r_test)))
		train_set = Dataset_train(
			p_idx=perdiadata.p_train, p_idx_len=perdiadata.p_len_train, r_idx=perdiadata.r_train, r_idx_len=perdiadata.r_len_train, r_history_idx=perdiadata.history_r_train,
			r_history_idx_pos=perdiadata.history_r_pos_train, r_mask=perdiadata.r_mask_train, r_mask_pos=perdiadata.r_mask_post_train, p_pos=perdiadata.p_train_pos, 
			p_history_idx=perdiadata.history_p_train, p_history_idx_pos=perdiadata.history_p_pos_train)

		train_sampler = dist.DistributedSampler(train_set, num_replicas=hvd.size(), rank=hvd.rank()) \
							if multi_gpu else None
		train_loader = DataLoader(train_set,
		batch_size=opt.batch_size,
		shuffle=False if multi_gpu else True,
		sampler=train_sampler,
		drop_last=True,
		collate_fn=collate_fn_train)
		# trans_data = TranslateData(pad_id)
		dev_set = Dataset_train(
			p_idx=perdiadata.p_dev, p_idx_len=perdiadata.p_len_dev, r_idx=perdiadata.r_dev, r_idx_len=perdiadata.r_len_dev, r_history_idx=perdiadata.history_r_dev,
			r_history_idx_pos=perdiadata.history_r_pos_dev, r_mask=perdiadata.r_mask_dev, r_mask_pos=perdiadata.r_mask_post_dev,p_pos=perdiadata.p_dev_pos, 
			p_history_idx=perdiadata.history_p_dev, p_history_idx_pos = perdiadata.history_p_pos_dev)
		dev_sampler = dist.DistributedSampler(dev_set, num_replicas=hvd.size(), rank=hvd.rank()) \
							if multi_gpu else None
		dev_loader = DataLoader(dev_set,
		batch_size=opt.batch_size,
		shuffle=False if multi_gpu else True,
		sampler=dev_sampler,
		collate_fn=collate_fn_train)
		# Prepare optimizer
		optimizer = Optimizer(optim.Adam(seq2seq.parameters(), lr=opt.learning_rate), max_grad_norm=opt.clip_grad)
		optimizer = optim.Adam(seq2seq.parameters(), lr=opt.learning_rate)
		if multi_gpu: 
			optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=seq2seq.named_parameters())
			hvd.broadcast_optimizer_state(optimizer, root_rank=0)
			hvd.broadcast_parameters(seq2seq.state_dict(), root_rank=0)
		optimizer = Optimizer(optimizer, max_grad_norm=opt.clip_grad)
		if opt.decay_factor:
			optimizer.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'min', factor=opt.decay_factor, patience=1))
		# Prepare trainer and train
		t = SupervisedTrainer(loss=loss, 
							  model_dir=opt.model_dir,
							  best_model_dir=opt.best_model_dir,
							  batch_size=opt.batch_size,
							  checkpoint_every=opt.checkpoint_every,
							  print_every=opt.print_every,
							  max_epochs=opt.max_epochs,
							  max_steps=opt.max_steps,
							  max_checkpoints_num=opt.max_checkpoints_num,
							  best_ppl=opt.best_ppl,
							  device=device,
							  multi_gpu=multi_gpu,
							  logger=logger,src_idx2word=src_vocab.idx2word,tgt_idx2word=tgt_vocab.idx2word,tgt_vocab=tgt_vocab)
		seq2seq = t.train(seq2seq, 
						  data=train_loader,
						  start_step=opt.skip_steps, 
						  dev_data=dev_loader,
						  optimizer=optimizer,
						  teacher_forcing_ratio=opt.teacher_forcing_ratio)
	if opt.phase == "train":
		predictor = PerPredictor(seq2seq, src_vocab.word2idx, tgt_vocab.idx2word, device)
		cnt = 0
		p, p_len, r, r_len, history_r, history_r_pos, r_mask, r_mask_post, history_p, history_p_pos, p_pos = test_data
		fw = open(opt.result_path,"w")
		lines = []
		for i in tqdm(range(len(test_data[0]))):
			p_idx = p[i]
			r_idx = r[i]
			p_len_i = p_len[i]
			r_len_i = r_len[i] 
			p_txt = [ src_vocab.idx2word[pw]  for pw in p_idx]
			r_txt = [ tgt_vocab.idx2word[pw] for pw in r_idx] 
			ans = predictor.predict_n(test_data,i=i,n=opt.beam_width) \
		        if opt.beam_width > 1 else predictor.predict(test_data,i=i)
			history_txts = []
			history_r_i = history_r[i]
			history_r_i_len = len(history_r_i)
			harange = np.random.permutation(history_r_i_len)
			selected_history_len = 5
			if history_r_i_len < 5:
				selected_history_len = history_r_i_len
			harange = harange[:selected_history_len]
			for idx in range(len(history_r_i)):
				history_r_idx_txt =  [tgt_vocab.idx2word[w] for w in history_r_i[idx]]
				history_txts.append(history_r_idx_txt)
			lines.append(json.dumps({"post":p_txt,"answer":r_txt,"history":history_txts,"result":ans},ensure_ascii=False))
			if i > 1200:
				break
		fw.write('\n'.join(lines))
		fw.close()
