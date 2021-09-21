from collections import defaultdict
import numpy as np

class VocabField():
    def __init__(self, vocab, vocab_size=None, unk_token="<UNK>", pad_token="<PAD>", sos_token=None, eos_token=None):
        default_tokens = [unk_token, pad_token]
        if sos_token: default_tokens.append(sos_token)
        if eos_token: default_tokens.append(eos_token)
        default_tokens.append('<MASK>')
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        vocab = default_tokens + vocab
        if vocab_size: vocab = vocab[:vocab_size]
        unk_id = vocab.index(unk_token)
        self.vocab = vocab
        self.word2idx = defaultdict(lambda : unk_id)
        self.idx2word = defaultdict(lambda : unk_token)
        for i, w in enumerate(vocab):
            self.word2idx[w] = i
            self.idx2word[i] = w
        self.vocab_size = len(self.word2idx)

    @staticmethod
    def load_vocab(vocab_fp):
        vocab = []
        with open(vocab_fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line: vocab.append(line)
        return vocab
    
    @staticmethod
    def load_from_pretrained(vocab_fp):
        f = open(vocab_fp,'r')
        vecs = []
        vocab = []
        word_num = 0
        for i, line in enumerate(f):
            if i == 0:
                continue
            data = line.split()
            word = data[0]
            vec = [float(d) for d in data[1:]]
            if len(vocab) > 50000:
                break
            assert len(vec) == 100
            vecs.append(vec)
            vocab.append(word)
            word_num += 1

        embedding = np.random.rand(word_num + 5, 100)
        for i,vec in enumerate(vecs):
            embedding[i+5] = vec
        return vocab, embedding
