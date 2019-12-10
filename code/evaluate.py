# -*- coding: UTF-8 -*-

import torch
from dataset import word_ids2string
from evalm import distance
import logging


def test(transducer, test_iter, beam_width=4, output_file=None, cuda=False, verbose=False):
    '''

    :param transducer: model
    :param test_iter: test dataloader @NOTE: must not shuffle
    :param cuda: cuda
    :param verbose:
    :return:
    '''
    ed = 0.0
    acc = 0.0
    count = 0.0
    out_list = []

    transducer.eval()

    if cuda:
        transducer.cuda()
    if verbose:
        logger = logging.getLogger()

    test_dataset = test_iter.dataset

    data_id = 0
    with torch.no_grad():
        for lemmas, lemma_lens, _, word_lens, feats, poss, m_lemmas, m_words in test_iter:
            '''
            lemmas, lemma_lens, feats, poss: model input
            words: label 2d Tensor (batchsize * padded_len)
            word_lens: label length before padding
            '''
            if cuda:
                lemmas, lemma_lens, feats, poss = lemmas.cuda(), lemma_lens.cuda(), feats.cuda(), poss.cuda()
            _, prediction, _ = transducer(lemmas, lemma_lens, feats, poss, m_lemmas, beam_width=beam_width)
            # prediction:  2d list (batch * various length)
            _words = m_words.tolist()
            label = [_words[i][:word_lens[i]] for i in range(len(_words))]
            count += len(label)
            for i in range(len(label)):
                ed += distance(label[i], prediction[i])
                acc += (label[i] == prediction[i])
                if output_file is not None or verbose:
                    #lemma_str = ''.join(char_vocab.i2w(j.item()) for j in lemmas[i])
                    lemma_str = ''.join(test_dataset.raw_data[data_id][0])
                    #pred_str = ''.join(char_vocab.i2w(j) for j in prediction[i])
                    pred_str = ''.join(word_ids2string(prediction[i], test_iter.dataset.get_m_voacb()))
                    # poss[i] can be a one hot vector or an integer
                    #pos = pos_vocab.i2w(sum(poss[i]).item())
                    #feat = [feat_vocab.i2w(j.item()) for j in feats[i] if j.item() != feat_vocab.unk]
                    #ft_str = ';'.join([pos] + feat)
                    ft_str = ';'.join(test_dataset.raw_data[data_id][3] + test_dataset.raw_data[data_id][2])
                    out_list.append('\t'.join([lemma_str, pred_str, ft_str]))
                if verbose:
                    logger.info(str(label[i]))
                    logger.info(prediction[i])
                    logger.info(out_list[-1])
                data_id += 1

    if output_file is not None:
        with open(output_file, 'w') as f:
            for item in out_list:
                f.write("%s\n" % item)

    ed = ed / count
    acc = acc / count

    return {'acc': acc, 'ed': ed}

