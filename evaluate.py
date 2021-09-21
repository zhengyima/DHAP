import os,json,logging,argparse,re,gensim
from rouge import Rouge
from tqdm import tqdm 
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.translate.bleu_score import sentence_bleu
from itertools import chain
import numpy as np
import math

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
				 left_pad_symbol=None, right_pad_symbol=None):

	sequence = iter(sequence)
	if pad_left:
		sequence = chain((left_pad_symbol,) * (n - 1), sequence)
	if pad_right:
		sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
	return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
		   left_pad_symbol=None, right_pad_symbol=None):

	sequence = pad_sequence(sequence, n, pad_left, pad_right,
							left_pad_symbol, right_pad_symbol)

	history = []
	while n > 1:
		try:
			next_item = next(sequence)
		except StopIteration:
			return
		history.append(next_item)
		n -= 1
	for item in sequence:
		history.append(item)
		yield tuple(history)
		del history[0]

def distinct_n_sentence_level(sentence, n):
	if len(sentence) == 0:
		return 0.0  # Prevent a zero division
	distinct_ngrams = set(ngrams(sentence, n))
	return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
	# return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
	sentencelist = []
	for s in sentences:
		sentencelist.extend(s)
	return distinct_n_sentence_level(sentencelist,n)
def ppl(textTest,train,n_gram=4):
	n = n_gram
	tokenized_text = [list(map(str.lower, sent)) 
					for sent in train]
	train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

	tokenized_text = [list(map(str.lower, sent)) 
					for sent in textTest]
	test_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

	model = Laplace(1) 
	model.fit(train_data, padded_sents)

	s = 0
	for i, test in enumerate(test_data):
		p = model.perplexity(test)
		s += p
	return s / (i + 1)
def rouge(candidate, reference):
	'''
	f:F1值  p：查准率  R：召回率
	a = ["i am a student from china"]  # 预测摘要 （可以是列表也可以是句子）
	b = ["i am student from school on japan"] #真实摘要
	'''
	rouge = Rouge()
	rouge_score = rouge.get_scores(" ".join(candidate), " ".join(reference))
	return rouge_score[0]["rouge-1"]['r'], rouge_score[0]["rouge-2"]['r'], rouge_score[0]["rouge-l"]['r']

def preprocess_result(filepath):
	f = open(filepath)
	test_path = "test.eval"
	infer_path = "infer.eval"
	f1 = open(test_path,"w")
	f2 = open(infer_path,"w")
	for line in f:
		r = json.loads(line.strip())
		p = r['post']
		a = r['answer']
		res = r['result']

		a_str = ' '.join(a)
		r_str = ' '.join(res)
		a_str = a_str.replace("<SOS>","").replace("<EOS>","").replace("<PAD>","").replace("<UNK>","")
		r_str = r_str.replace("<SOS>","").replace("<EOS>","").replace("<PAD>","").replace("<UNK>","")
		if len(r_str.strip()) == 0:
			continue
		if len(a_str.strip()) == 0:
			continue
		f1.write(a_str+"\n")
		f2.write(r_str+"\n")
	f.close()
	f1.close()
	f2.close()
	return test_path, infer_path


def cal_vector_extrema(x, y, dic):
	# x and y are the list of the words
	# dic is the gensim model which holds 300 the google news word2ved model
	def vecterize(p):
		vectors = []
		for w in p:
			if w in dic:
				vectors.append(dic[w.lower()])
		if not vectors:
			vectors.append(np.random.randn(300))
		return np.stack(vectors)
	x = vecterize(x)
	y = vecterize(y)
	vec_x = np.max(x, axis=0)
	vec_y = np.max(y, axis=0)
	assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
	zero_list = np.zeros(len(vec_x))
	if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
		return float(1) if vec_x.all() == vec_y.all() else float(0)
	res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
	cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
	return cos


def cal_embedding_average(x, y, dic):
	# x and y are the list of the words
	def vecterize(p):
		vectors = []
		for w in p:
			if w in dic:
				vectors.append(dic[w.lower()])
		if not vectors:
			vectors.append(np.random.randn(300))
		return np.stack(vectors)
	x = vecterize(x)
	y = vecterize(y)
	
	vec_x = np.array([0 for _ in range(len(x[0]))])
	for x_v in x:
		x_v = np.array(x_v)
		vec_x = np.add(x_v, vec_x)
	vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
	vec_y = np.array([0 for _ in range(len(y[0]))])
	for y_v in y:
		y_v = np.array(y_v)
		vec_y = np.add(y_v, vec_y)
	vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
	assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
	zero_list = np.array([0 for _ in range(len(vec_x))])
	if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
		return float(1) if vec_x.all() == vec_y.all() else float(0)
	vec_x = np.mat(vec_x)
	vec_y = np.mat(vec_y)
	num = float(vec_x * vec_y.T)
	denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
	cos = num / denom
	return cos

def cal_greedy_matching(x, y, dic):
	# x and y are the list of words
	def vecterize(p):
		vectors = []
		for w in p:
			if w in dic:
				vectors.append(dic[w.lower()])
		if not vectors:
			vectors.append(np.random.randn(300))
		return np.stack(vectors)
	x = vecterize(x)
	y = vecterize(y)
	len_x = len(x)
	len_y = len(y)
	cosine = []
	sum_x = 0 
	for x_v in x:
		for y_v in y:
			assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
			zero_list = np.zeros(len(x_v))
			if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
				if x_v.all() == y_v.all():
					cos = float(1)
				else:
					cos = float(0)
			else:
				# method 1
				res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
				cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

			cosine.append(cos)
		if cosine:
			sum_x += max(cosine)
			cosine = []

	sum_x = sum_x / len_x
	cosine = []

	sum_y = 0

	for y_v in y:

		for x_v in x:
			assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
			zero_list = np.zeros(len(y_v))

			if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
				if (x_v == y_v).all():
					cos = float(1)
				else:
					cos = float(0)
			else:
				# method 1
				res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
				cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

			cosine.append(cos)

		if cosine:
			sum_y += max(cosine)
			cosine = []

	sum_y = sum_y / len_y
	score = (sum_x + sum_y) / 2
	return score


def cal_greedy_matching_matrix(x, y, dic):
	# x and y are the list of words
	def vecterize(p):
		vectors = []
		for w in p:
			if w in dic:
				vectors.append(dic[w.lower()])
		if not vectors:
			vectors.append(np.random.randn(300))
		return np.stack(vectors)
	x = vecterize(x)     # [x, 300]
	y = vecterize(y)     # [y, 300]
	len_x = len(x)
	len_y = len(y)
	matrix = np.dot(x, y.T)    # [x, y]
	matrix = matrix / np.linalg.norm(x, axis=1, keepdims=True)    # [x, 1]
	matrix = matrix / np.linalg.norm(y, axis=1).reshape(1, -1)    # [1, y]
	x_matrix_max = np.mean(np.max(matrix, axis=1))    # [x]
	y_matrix_max = np.mean(np.max(matrix, axis=0))    # [y]
	return (x_matrix_max + y_matrix_max) / 2

def cal_s_for_each_history(r, h, idf_dict):
	c = 0
	has_c = {}
	for w in r:
		if w in h and w not in has_c:
			c += idf_dict[w]
			has_c[w] = 1
	return c

def docs(w, history_list):
	c = 0
	for i,h in enumerate(history_list):
		if w in h:
			c += 1
	return c

def gen_idf_dict(history_list):
	idf_dict = {}
	for i, h in enumerate(history_list):
		for w in h:
			if w not in idf_dict:
				idf = math.log(len(history_list) *1.0 / docs(w, history_list))
				idf_dict[w] = idf

	return idf_dict
	
def cal_p_cover(file_name):
	s_sum = 0
	with open(file_name) as f:
		line_cnt = 0
		for line in f:
			line_dic = json.loads(line.strip())
			result = line_dic['result']
			history = line_dic['history']
			idf_dict = gen_idf_dict(history)
			a1 = sorted(idf_dict.items(),key = lambda x:x[1],reverse = True)
			# print(a1)
			s_list = []
			for i, h in enumerate(history):
				h = ' '.join(h).replace("<PAD>","").replace("<EOS>","").replace("<SOS>","").split()
				r = ' '.join(result).replace("<EOS>","").split()
				s = cal_s_for_each_history(r, h, idf_dict)
				s_list.append(s)
			s_max = max(s_list)
			s_sum += s_max
			line_cnt += 1
	return (s_sum+0.0)/line_cnt
	
parser = argparse.ArgumentParser()
parser.add_argument('--result_path', action='store', dest='result_path', default='./res.txt')
parser.add_argument('--emb_path', action='store', dest='emb_path', default='./res.txt')

opt = parser.parse_args()

if __name__ == '__main__':

	tp, ip = preprocess_result(opt.result_path)
	p_cover = cal_p_cover(opt.result_path)
	print("p_cover", p_cover)
	with open(opt.result_path) as f:
		ref, tgt = [], []
		for idx, line in enumerate(f.readlines()):
			line_dic = json.loads(line.strip())
			ref.append(' '.join(line_dic['answer']).replace("<UNK>","").split())
			tgt.append(' '.join(line_dic['result']).replace("<UNK>","").split())
	assert len(ref) == len(tgt)
	dic = gensim.models.KeyedVectors.load_word2vec_format(opt.emb_path, binary=False)
	print('[!] load the word2vector by gensim over')
	ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
	no_save = 0
	for rr, cc in tqdm(list(zip(ref, tgt))):
		ea_sum_ = cal_embedding_average(rr, cc, dic)
		vx_sum_ = cal_vector_extrema(rr, cc, dic)
		gm_sum += cal_greedy_matching_matrix(rr, cc, dic)
		# gm_sum += cal_greedy_matching(rr, cc, dic)
		if ea_sum_ != 1 and vx_sum_ != 1:
			ea_sum += ea_sum_
			vx_sum += vx_sum_
			counterp += 1
		else:
			no_save += 1    
	if not os.path.exists('logging'):
		os.mkdir('logging')
	logging_fp = 'logging/' + ip + '.log'
	log_formatter = logging.Formatter('%(message)s')
	log_handler = logging.FileHandler(logging_fp)
	log_handler.setFormatter(log_formatter)
	logger = logging.getLogger('eval')
	logger.addHandler(log_handler)
	logger.setLevel(level=logging.INFO)
	corpus_train = tp
	text = []
	num = 0
	idx = 0
	out_file = open(ip,'r')
	candidate = []
	bleu_score_all_1 = 0
	bleu_score_all_2 = 0
	bleu_score_all_3 = 0
	bleu_score_all_4 = 0
	rouge_score_all_1 = 0
	rouge_score_all_2 = 0
	rouge_score_all_l = 0
	train_sentence = []
	for line in out_file:
		if False:
			r = seg.cut(''.join(line.strip()))
		else:
			r = line.strip().split(' ')
		candidate.append(r)
	with open(corpus_train,'r') as f:
		for idx, line in tqdm(enumerate(f.readlines())):
			reference = []
			data = line
			resps_num = len(data)
			for resp in data.strip().split('\t'):
				reference.append(resp.split(' '))
				train_sentence.append(resp.split(' '))
			bleu_score_1 = sentence_bleu(reference, candidate[idx],weights=(1, 0, 0, 0))
			bleu_score_all_1 += bleu_score_1
			bleu_score_2 = sentence_bleu(reference, candidate[idx],weights=(0.5, 0.5, 0, 0))
			bleu_score_all_2 += bleu_score_2
			bleu_score_3 = sentence_bleu(reference, candidate[idx],weights=(0.33, 0.33, 0.33, 0))
			bleu_score_all_3 += bleu_score_3
			bleu_score_4 = sentence_bleu(reference, candidate[idx],weights=(0.25, 0.25, 0.25, 0.25))
			bleu_score_all_4 += bleu_score_4
			rouge_score_1, rouge_score_2, rouge_score_l = rouge(candidate[idx], reference[0])
			rouge_score_all_1 += rouge_score_1
			rouge_score_all_2 += rouge_score_2
			rouge_score_all_l += rouge_score_l
			num += 1
	# ppl_score_1 = ppl(candidate,train_sentence,1)
	# ppl_score_2 = ppl(candidate,train_sentence,2)
	distinct_score_1 = distinct_n_corpus_level(candidate,1)
	distinct_score_2 = distinct_n_corpus_level(candidate,2)
	# logger.info('BLEU-1:%f, BLEU-2:%f,BLEU-3:%f,BLEU-4:%f,DISTINCT-1:%f,DISTINCT-2:%f, ROUGE-1:%f, ROUGE-2:%f, ROUGE-L:%f',
	#     bleu_score_all_1 / num, bleu_score_all_2 / num, bleu_score_all_3 / num, bleu_score_all_4 / num,
	#     distinct_score_1,distinct_score_2, rouge_score_all_1 / num, rouge_score_all_2/ num, rouge_score_all_l / num)
	print('BLEU-1:%f, BLEU-2:%f,BLEU-3:%f,BLEU-4:%f,DISTINCT-1:%f,DISTINCT-2:%f, ROUGE-1:%f, ROUGE-2:%f, ROUGE-L:%f' % (
		bleu_score_all_1 / num, bleu_score_all_2 / num, bleu_score_all_3 / num, bleu_score_all_4 / num,
		distinct_score_1,distinct_score_2, rouge_score_all_1 / num, rouge_score_all_2/ num, rouge_score_all_l / num))
	print(f'EA: {round(ea_sum / counterp, 4)}')
	print(f'VX: {round(vx_sum / counterp, 4)}')
	print(f'GM: {round(gm_sum / counterp, 4)}')
	out_file.close()
