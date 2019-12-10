import sys
import numpy as np
import dynet as dy
sys.path.append("..")
from vocab import Vocabulary
from evalm import distance

MAX_ACTION_NUM = 150

COPY = '<COPY>'
DELETE = '<DEL>'

class Transducer(object):
    def __init__(self, model, vocab_char, vocab_feat, vocab_pos, 
                 c_emb_dim=100, a_emb_dim=100, f_emb_dim=20, 
                 encoder_hidden_dim=200, encoder_layer_num=1, 
                 decoder_hidden_dim=200, decoder_layer_num=1, rnn_type='lstm', 
                 ac_share_emb=True, pos_sp=True):
        """Transducer

        @ TODO: consider current generated char in decoding

        Args:
            vocab_char: lemma and target word character vocabulary
            vocab_feat: grammar feature vocabulary
            vocab_pos: POS tags vocabulary
            c_emb_dim: char embedding dim
            a_emb_dim: action embedding dim
            f_emb_dim: feature & pos embedding dim
            encoder_hidden_dim: encoder rnn hidden size
            encoder_layer_num: number of encoder rnn layers
            decoder_hidden_dim: decoder rnn hidden size
            decoder_layer_num: number of decoder rnn layers
            rnn_type: lstm | coupled_lstm | gru
            ac_share_emb: whether the acts (inserts) and the chars share embeddings
            pos_sp: whether specially treat pos tags
        """
        self.vocab_char = vocab_char
        self.vocab_feat = vocab_feat
        self.vocab_pos = vocab_pos
        self.c_emb_dim = c_emb_dim
        self.a_emb_dim = a_emb_dim
        self.f_emb_dim = f_emb_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_layer_num = encoder_layer_num
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_layer_num = decoder_layer_num
        self.rnn_type = rnn_type
        self.ac_share_emb = ac_share_emb
        self.pos_sp = pos_sp
        
        # Check param
        if self.ac_share_emb:
            assert self.a_emb_dim == self.c_emb_dim

        # Build action vocabulary
        self.vocab_act = Vocabulary.from_vocab(self.vocab_char)
        self.vocab_act.add(COPY, special=True)
        self.copy = self.vocab_act.w2i(COPY)
        self.vocab_act.add(DELETE, special=True)
        self.delete = self.vocab_act.w2i(DELETE)

        # Embeddings
        if self.ac_share_emb:
            self.embedding_char = model.add_lookup_parameters((len(self.vocab_act), self.c_emb_dim))
            self.embedding_act = self.embedding_char
        else:
            self.embedding_char = model.add_lookup_parameters((len(self.vocab_char), self.c_emb_dim))
            self.embedding_act = model.add_lookup_parameters((len(self.vocab_act), self.a_emb_dim))
        self.embedding_feat = model.add_lookup_parameters((len(self.vocab_feat), self.f_emb_dim))
        self.embedding_pos = model.add_lookup_parameters((len(self.vocab_pos), self.f_emb_dim))