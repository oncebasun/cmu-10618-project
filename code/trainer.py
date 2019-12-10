# -*- coding: UTF-8 -*-
import os
import logging
import torch
import numpy as np

from utils import save_model
from evaluate import test


class Trainer(object):
    default_adam_args = {"lr": 0.001,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    default_adadelta_args = {"lr": 1.0,
                             "rho": 0.95,
                             "eps": 1e-6,
                             "weight_decay": 0.0}

    def __init__(self, optim="adadelta", optim_args={}):
        if optim == "adadelta":
            optim_args_merged = self.default_adadelta_args.copy()
            optim_args_merged.update(optim_args)
            self.optim_args = optim_args_merged
            self.optim = torch.optim.Adadelta
        elif optim == "adam":
            optim_args_merged = self.default_adam_args.copy()
            optim_args_merged.update(optim_args)
            self.optim_args = optim_args_merged
            self.optim = torch.optim.Adam
        else:
            raise NotImplementedError

        self.logger = logging.getLogger()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.dev_acc_history = []

    def train(self, transducer, train_iter, dev_iter, num_epochs=60, 
              patience=20, roll_in_k=12, roll_out_p=0.5, beam_width=4, 
              clip=10.0, l2=0.0, cuda=False, best=True, model_dir='../model/', 
              verbose=False):
        """
        @TODO: time
        """
        # Zero gradients of both optimizers
        optim = self.optim(transducer.parameters(), **self.optim_args)

        self._reset_histories()

        if cuda:
            transducer.cuda()
        
        self.logger.info('START TRAIN')
        self.logger.info('CUDA = ' + str(cuda))

        save_model(model_dir, transducer)

        best_dev_acc = 0.0
        best_dev_ed = float("inf")
        best_epoch = 0

        epoch = 1
        while epoch <= num_epochs:
            transducer.train()

            epoch_loss = 0.0

            model_roll_in_p = 1 - (roll_in_k / (roll_in_k + np.exp(float(epoch)/roll_in_k)))
            
            self.logger.info('Epoch: %d/%d start ... model_roll_in_p: %f' % (epoch, num_epochs, model_roll_in_p))

            for ss, batch in enumerate(train_iter):
                lemma, lemma_len, word, word_len, feat, pos, m_lemma, _ = batch

                # Reset
                optim.zero_grad()
                loss = 0

                if cuda:
                    lemma = lemma.cuda()
                    lemma_len = lemma_len.cuda()
                    word = word.cuda()
                    word_len = word_len.cuda()
                    feat = feat.cuda()
                    pos = pos.cuda()
                    m_lemma = m_lemma.cuda()

                # Run batch through transducer
                loss, prediction, predicted_acts = transducer(lemma, lemma_len, feat, pos, m_lemma, word, word_len, model_roll_in_p=model_roll_in_p)

                # L2 Regularization
                l2_reg = None
                for W in transducer.parameters():  #@BUG: do not include embeddings 
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)

                # update loss
                loss = loss + l2_reg * l2

                epoch_loss += loss.item()

                # Backpropagation
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(transducer.parameters(), clip)
                optim.step()

                '''
                @TODO: tf board
                info = {
                    'Loss': loss.data[0],
                }
                for tag, value in info.items():
                    self.tf_logger.scalar_summary(tag, value, ss)
                '''

                if verbose:
                    self.logger.info('Epoch: %d/%d, Iteration: %d, loss: %f' % (epoch, num_epochs, ss, loss.item()))
                

            self.logger.info('  ave train loss: %f' % (epoch_loss/len(train_iter)))

            dev_scores = test(transducer, dev_iter, beam_width=beam_width, 
                              output_file= None, cuda=cuda, verbose=verbose)
            #dev_scores = {'acc':0.0, 'ed':0.0}
            if dev_scores['acc'] > best_dev_acc:
                best_dev_acc = dev_scores['acc']
                best_dev_ed = dev_scores['ed']
                best_epoch = epoch
                num_epochs = max(epoch + patience, num_epochs)
            self.logger.info('  current dev acc: %f, ed: %f | highest dev acc: %f, ed: %f @ epoch %d' % (dev_scores['acc'], dev_scores['ed'], best_dev_acc, best_dev_ed, best_epoch))

            if best :
                if epoch == best_epoch:
                    save_model(model_dir, transducer)
            else:
                save_model(model_dir, transducer)
                
            epoch += 1
