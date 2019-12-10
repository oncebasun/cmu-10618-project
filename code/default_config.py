# -*- coding: UTF-8 -*-
def get_default_config():
    args = lambda: None

    # Data path
    args.datadir = '../data'
    # Incorporate validation data into vocabulary
    args.incorp_val = False
    # Random seed
    args.seed = None
    # Initial learning rate
    args.lr = 0.001
    # Shuffle the data every epoch
    args.shuffle = True
    # Store model of the best epoch
    args.best = True
    # Fix the embeddings 
    args.static = False
    # Specially treat pos tags
    args.pos_sp = False
    # Use existing word embeddings
    args.emb = None
    # Char embedding dim
    args.c_emb_dim = 100
    # Action embedding dim
    args.a_emb_dim = 100
    # Feature embedding dim
    args.f_emb_dim = 20
    # Encoder RNN hidden size
    args.encoder_hidden_dim = 200
    # Decoder RNN hidden size
    args.decoder_hidden_dim = 200
    # Number of encoder RNN layers
    args.encoder_layer_num = 1
    # Number of decoder RNN layers
    args.decoder_layer_num = 1
    # lstm | coupled_lstm | gru
    args.rnn_type = 'lstm'
    # Whether the acts (inserts) and the chars share embeddings
    args.ac_share_emb = False
    # Batch size
    args.batch_size = 1
    # Beam size
    args.beam_width = 4
    # Epochs
    args.epochs = 60
    # Patience
    args.patience = 20
    # Clips gradient norm of an iterable of parameters
    args.clip = 10.0
    # K for roll in sampling
    args.roll_in_k = 12
    # Probability threshold for roll-out sampling
    args.roll_out_p = 0.5
    # L2 Regularization scale
    args.l2 = 0.0
    # adadelta | adam
    args.optim = 'adadelta'


    return args