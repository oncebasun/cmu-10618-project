import copy
import pickle


UNK = '<UNK>'
PAD = '<PAD>'
BOS = '<s>'
EOS = '<\s>'


class Vocabulary(object):
    def __init__(self, unk=True, pad=False, bos=False, eos=False):
        self.has_unk = unk
        self.has_pad = pad
        self.has_bos = bos
        self.has_eos = eos
        self.dic_w2i = {}
        self.dic_i2w = {}

        self.ids = []
        self.unspecial_ids = []

        if self.has_unk:
            self.add(UNK, special=True)
            self.unk = self.w2i(UNK)
        if self.has_pad:
            self.add(PAD, special=True)
            self.pad = self.w2i(PAD)
        if self.has_bos:
            self.add(BOS, special=True)
            self.bos = self.w2i(BOS)
        if self.has_eos:
            self.add(EOS, special=True)
            self.eos = self.w2i(EOS)
        
    def add(self, word, special=False):
        if word not in self.dic_w2i:
            self.dic_w2i[word] = len(self.dic_w2i)
            if not special:
                self.unspecial_ids.append(self.dic_w2i[word])
            self.ids.append(self.dic_w2i[word])
            self.dic_i2w[self.dic_w2i[word]] = word

    def encode(self, sequence, growth=False):
        ret = []
        for word in sequence:
            if word not in self.dic_w2i:
                if growth:
                    self.add(word)
            ret.append(self.dic_w2i.get(word, self.unk))
        if self.has_bos:
            ret = [self.bos] + ret
        if self.has_eos:
            ret = ret + [self.eos]
        return ret

    def decode(self, ids):
        return map(lambda idx: self.i2w(idx), ids)

    def w2i(self, word):
        return self.dic_w2i[word]

    def i2w(self, index):
        return self.dic_i2w[index]

    def copy(self, vocab):
        """copy from existing vocabulary"""
        self.dic_w2i = copy.deepcopy(vocab.dic_w2i)
        self.dic_i2w = copy.deepcopy(vocab.dic_i2w)
        self.has_unk = vocab.has_unk
        self.has_pad = vocab.has_pad
        self.has_bos = vocab.has_bos
        self.has_eos = vocab.has_eos
        if self.has_unk:
            self.unk = self.w2i(UNK)
        if self.has_pad:
            self.pad = self.w2i(PAD)
        if self.has_bos:
            self.bos = self.w2i(BOS)
        if self.has_eos:
            self.eos = self.w2i(EOS)
        self.unspecial_ids = list(vocab.unspecial_ids)
        self.ids = list(vocab.ids)
        
    def save(self, save_file):
        with open(save_file, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, save_file):
        vocab = cls()
        with open(save_file, 'rb') as fp:
            saved_vocab = pickle.load(fp)
            vocab.copy(saved_vocab)
        return vocab

    @classmethod
    def from_vocab(cls, vocab):
        ret_vocab = cls()
        ret_vocab.copy(vocab)
        return ret_vocab

    def __len__(self):
        assert len(self.dic_w2i) == len(self.dic_i2w)
        return len(self.dic_w2i)
