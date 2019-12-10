# -*- coding: UTF-8 -*-
import os
import time
import random
import pickle
import logging
import datetime
import argparse
import importlib
import numpy as np

import torch
import default_config
import configs.empty_config as configuration
import utils

from model.transducer import Transducer
from dataset import MorphDataset, MorphDataloader
from vocab import Vocabulary
from trainer import Trainer
from evaluate import test

if __name__ == '__main__':
    ###############################################
    #                 Preparation                 #
    ###############################################
    # Configuration priority 0 > 1 > 2 > 3 > 4 > 5:
    # 0. (only for model configs) loaded model config
    # 1. command line options
    # 2. default command line options
    # 3. command line config file 
    # 4. main config file 
    # 5. defult 

    args = default_config.get_default_config()
    configuration.update_config(args)

    parser = argparse.ArgumentParser(description='Reimplementation of "Peter Makarov and Simon Clematide. 2018. Uzh at conll-sigmorphon 2018 shared task on universal morphological reinflection".')
    # options
    parser.add_argument('--config', type=str, metavar='CONFIG', help='use this configuration file instead of the default config, like "configs.empty_config"')
    parser.add_argument('--test', action='store_true', default=False, help='train | test')
    parser.add_argument('--output_pred', action='store_true', default=False, help='whether output prediction to file')
    parser.add_argument('--load', type=str, default=None, help='dir of model to load [default: None]')
    parser.add_argument('--debug', action='store_true', default=False, help='show DEBUG outputs')
    parser.add_argument('--verbose', action='store_true', default=False, help='show more detailed output')
    parser.add_argument('--modeldir', type=str, help='model saving dir')
    # device
    parser.add_argument('--device', type=str, default='cpu', help='device to use for iterate data. cpu | cudaX (e.g. cuda0) [default: cpu]')
    # data
    parser.add_argument('--lang', type=str, metavar='LANG', required=True, help='language')
    parser.add_argument('--split', type=str, metavar='SPLIT', required=True, help='training split (high | medium | low)')
    parser.add_argument('--datadir', '-d', type=str, help='path to the data directory [default: %s]' % args.datadir)
    parser.add_argument('--incorp_val', action='store_true', help='incorporate validation data into vocabulary [default: %s]' % getattr(args, "incorp_val"))
    #parser.add_argument('--emb', type=str, help='use existing word embeddings [default: %s]' %str(args.emb))
    # model
    parser.add_argument('--seed', type=int, help='manual seed [default: random seed or from config file]')
    parser.add_argument('--c_emb_dim', type=int, help='char embedding dim size [default: %d]' %args.c_emb_dim)
    parser.add_argument('--a_emb_dim', type=int, help='action embedding dim size [default: %d]' %args.a_emb_dim)
    parser.add_argument('--f_emb_dim', type=int, help='feature embedding dim size [default: %d]' %args.f_emb_dim)
    parser.add_argument('--encoder_hidden_dim', type=int, help='encoder RNN hidden size [default: %d]' %args.encoder_hidden_dim)
    parser.add_argument('--decoder_hidden_dim', type=int, help='decoder RNN hidden size [default: %d]' %args.decoder_hidden_dim)
    parser.add_argument('--encoder_layer_num', type=int, help='number of encoder RNN layers [default: %d]' %args.encoder_layer_num)
    parser.add_argument('--decoder_layer_num', type=int, help='number of decoder RNN layers [default: %d]' %args.decoder_layer_num)
    parser.add_argument('--rnn_type', type=str, help='lstm | coupled_lstm | gru [default: %s]' %args.rnn_type)
    parser.add_argument('--ac_share_emb', action='store_true', help='whether the acts (inserts) and the chars share embeddings [default: %s]' %str(args.ac_share_emb))
    parser.add_argument('--static', action='store_true', help='fix the embeddings [default: %s]' %str(args.static))
    parser.add_argument('--pos_sp', action='store_true', help='specially treat pos tags [default: %s]' %str(args.pos_sp))
    # training
    parser.add_argument('--lr', type=float, metavar='FLOAT', help='initial learning rate [default: %f]' % args.lr)
    parser.add_argument('--epochs', '-e', type=int, help='number of epochs for training [default: %d]' %args.epochs)
    parser.add_argument('--patience', type=int, help='patience [default: %d]' %args.patience)
    parser.add_argument('--batch_size', type=int, help='batch size for training [default: %d]' % args.batch_size)
    parser.add_argument('--best', action='store_true', help='store model of the best epoch [default: %s]' %str(args.best))
    parser.add_argument('--clip', type=float, help='clips gradient norm of an iterable of parameters [default: %f]' %args.clip)
    parser.add_argument('--roll_in_k', type=int, help='k for roll in sampling [default: %d]' %args.roll_in_k)
    parser.add_argument('--roll_out_p', type=float, help='probability threshold for roll-out sampling [default: %f]' %args.roll_out_p)
    parser.add_argument('--l2', type=float, help='l2 regularization scale [default: %f]' %args.l2)
    parser.add_argument('--optim', type=str, help='adadelta | adam [default: %s]' %args.optim)
    parser.add_argument('--beam_width', action='store_true', help='beam size in testing [default: %s]' %str(args.beam_width))
    new_args = parser.parse_args()

    # Update config
    if new_args.config is not None:
        new_config = importlib.import_module(new_args.config)
        new_config.update_config(args)
    for key in new_args.__dict__:
        if key is not 'config' and new_args.__dict__[key] is not None and not(key in args.__dict__ and isinstance(new_args.__dict__[key], bool) and new_args.__dict__[key] == False):
            setattr(args, key, new_args.__dict__[key])

    # Make up output paths
    utils.make_out_paths(args)

    # Loggers
    logger = utils.get_logger(args)
    logger.info('%%% Task start %%%')
    logger.debug('Logger is in DEBUG mode.')

    # Run mode
    if not args.test:
        logger.info('Running in TRAINING mode.')
        if hasattr(args, 'load') and args.load is not None:
            logger.info('Loading model: %s' % args.load)
            #utils.load_model_conf(args.load, args)
    elif args.test and args.load is not None:
        logger.info('Running in TESTING mode.')
        logger.info('Loading model: %s' % args.load)
        #utils.load_model_conf(args.load, args)
    elif args.test and (not hasattr(args, 'load') or args.load is None):
        logger.error('Running in Test mode and --load is not set')
        quit()

    # Device
    if args.device == 'cpu':
        args.cuda = False
        logger.info('Device: CPU')
    elif args.device[:4] == 'cuda':
        if torch.cuda.is_available():
            args.cuda = True
            gpuid = int(args.device[4:])
            torch.cuda.set_device(gpuid)
            logger.info('Device: CUDA #%d' % gpuid)
        else:
            args.cuda = False
            logger.warning('CUDA is not available now. Automatically switched to using CPU.')
    else:
        logging.error('Invalid device: %s !' % args.device)
        quit()

    # Seeding
    if args.seed is None:
        args.seed = random.randint(1, 100000000)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Show configurations
    logger.info('%Configuration%')
    for key in args.__dict__:
        logger.info('  %s: %s' %(key, str(args.__dict__[key])))

    # Save model configs
    if not args.test:
        logger.info('Saving configs ... ')
        #utils.save_model_conf(args.modeldir, args)

    # Load dataset
    if not args.test:  # train
        train_file = os.path.join(args.datadir, "%s-train-%s"%(args.lang, args.split))
        dev_file = os.path.join(args.datadir, "%s-dev"%(args.lang))
        test_file = os.path.join(args.datadir, "%s-test"%(args.lang))
        train_dataset = MorphDataset(train_file, pos_sp=args.pos_sp)
        char_vocab, feat_vocab, pos_vocab = train_dataset.get_vocabs()
        dev_dataset = MorphDataset(dev_file, char_vocab, feat_vocab, pos_vocab, 
                                   pos_sp=args.pos_sp, train=False)
        test_dataset = MorphDataset(test_file, char_vocab, feat_vocab, pos_vocab, 
                                    pos_sp=args.pos_sp, train=False)
        logging.info("Training data: " + os.path.abspath(train_file) + "  size: %d" %len(train_dataset))
        logging.info("Character vocabulary size: %d" %len(char_vocab))
        logging.info("Feature vocabulary size: %d" %len(feat_vocab))
        logging.info("POS vocabulary size: %d" %len(pos_vocab))
        logging.info("Dev data: " + os.path.abspath(dev_file) + "  size: %d" %len(dev_dataset))
        logging.info("Dev private character vocabulary size: %d" %len(dev_dataset.get_m_voacb()))
        logging.info("Test data: " + os.path.abspath(test_file) + "  size: %d" %len(test_dataset))
        logging.info("Test private character vocabulary size: %d" %len(test_dataset.get_m_voacb()))
        char_vocab.save(os.path.join(args.modeldir, 'char_vocab'))
        feat_vocab.save(os.path.join(args.modeldir, 'feat_vocab'))
        pos_vocab.save(os.path.join(args.modeldir, 'pos_vocab'))

        train_iter = MorphDataloader(train_dataset, left_padding=False, 
                                     batch_size=args.batch_size, shuffle=True, pin_memory=True)
        dev_iter = MorphDataloader(dev_dataset, left_padding=False, 
                                   batch_size=args.batch_size, shuffle=False)
        test_iter = MorphDataloader(test_dataset, left_padding=False, 
                                    batch_size=args.batch_size, shuffle=False)
    else:
        dev_file = os.path.join(args.datadir, "%s-dev"%(args.lang))
        test_file = os.path.join(args.datadir, "%s-test"%(args.lang))
        char_vocab = Vocabulary.load(os.path.join(args.load, 'char_vocab'))
        feat_vocab = Vocabulary.load(os.path.join(args.load, 'feat_vocab'))
        pos_vocab = Vocabulary.load(os.path.join(args.load, 'pos_vocab'))
        dev_dataset = MorphDataset(dev_file, char_vocab, feat_vocab, pos_vocab, 
                                   pos_sp=args.pos_sp, train=False)
        test_dataset = MorphDataset(test_file, char_vocab, feat_vocab, pos_vocab, 
                                    pos_sp=args.pos_sp, train=False)
        logging.info("Character vocabulary size: %d" %len(char_vocab))
        logging.info("Feature vocabulary size: %d" %len(feat_vocab))
        logging.info("POS vocabulary size: %d" %len(pos_vocab))
        logging.info("Dev data: " + os.path.abspath(dev_file) + "  size: %d" %len(dev_dataset))
        logging.info("Dev private character vocabulary size: %d" %len(dev_dataset.get_m_voacb()))
        logging.info("Test data: " + os.path.abspath(test_file) + "  size: %d" %len(test_dataset))
        logging.info("Test private character vocabulary size: %d" %len(test_dataset.get_m_voacb()))

        dev_iter = MorphDataloader(dev_dataset, left_padding=False, 
                                   batch_size=args.batch_size, shuffle=False)
        test_iter = MorphDataloader(test_dataset, left_padding=False, 
                                    batch_size=args.batch_size, shuffle=False)

    ###############################################
    ##            Constrcuting Models            ##
    ###############################################
    transducer = Transducer(char_vocab, feat_vocab, pos_vocab, 
                            args.c_emb_dim, args.a_emb_dim, args.f_emb_dim, 
                            args.encoder_hidden_dim, args.encoder_layer_num,
                            args.decoder_hidden_dim, args.decoder_layer_num,
                            args.rnn_type, args.ac_share_emb, args.pos_sp)
    if hasattr(args, 'load') and args.load is not None:
        utils.load_model(args.load, transducer)

    ###############################################
    ##                 Training                  ##
    ###############################################
    if not args.test:  # train
        trainer = Trainer(optim=args.optim, optim_args={'lr': args.lr})
        trainer.train(transducer, train_iter, dev_iter, args.epochs, 
                      args.patience, args.roll_in_k, args.roll_out_p, 
                      args.beam_width, args.clip, args.l2, cuda=args.cuda, 
                      best=args.best, model_dir=args.modeldir, 
                      verbose=args.verbose)

    ###############################################
    ##                 Predict                   ##
    ###############################################
    
    if not args.test:
        utils.load_model(args.modeldir, transducer)
    dev_output_file = None
    test_output_file = None
    if args.output_pred:
        dev_output_file = os.path.join(args.predout, 'predict_dev.txt')
        test_output_file = os.path.join(args.predout, 'predict_test.txt')

    dev_scores = test(transducer, dev_iter, beam_width=args.beam_width, 
                      output_file=dev_output_file, cuda=args.cuda, 
                      verbose=args.verbose)

    logger.info("dev acc: %f, ed: %f" % (dev_scores['acc'], dev_scores['ed']))
    
    test_scores = test(transducer, test_iter, beam_width=args.beam_width, 
                       output_file=test_output_file, cuda=args.cuda, 
                       verbose=args.verbose)

    logger.info("test acc: %f, ed: %f" % (test_scores['acc'], test_scores['ed']))
