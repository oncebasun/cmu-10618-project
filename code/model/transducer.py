# -*- coding: utf-8 -*-
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("..")
from vocab import Vocabulary
from lstm import CoupledLSTM
from evalm import distance

MAX_ACTION_NUM = 150

COPY = '<COPY>'
DELETE = '<DEL>'

class Transducer(nn.Module):
    def __init__(self, vocab_char, vocab_feat, vocab_pos, 
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
        super(Transducer, self).__init__()
        
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
            self.embedding_char = nn.Embedding(len(self.vocab_act), self.c_emb_dim, padding_idx=self.vocab_act.pad)
            self.embedding_act = self.embedding_char
        else:
            self.embedding_char = nn.Embedding(len(self.vocab_char), self.c_emb_dim, padding_idx=self.vocab_char.pad)
            self.embedding_act = nn.Embedding(len(self.vocab_act), self.a_emb_dim, padding_idx=self.vocab_act.pad)
        self.embedding_feat = nn.Embedding(len(self.vocab_feat), self.f_emb_dim)
        self.embedding_pos = nn.Embedding(len(self.vocab_pos), self.f_emb_dim)

        # Encoder
        if self.rnn_type == 'lstm':
            self.encoder = nn.LSTM(self.c_emb_dim, self.encoder_hidden_dim, 
                                   self.encoder_layer_num, bidirectional=True, 
                                   batch_first=True)
            # self.init_hidden = self._lstm_init_hidden
        elif self.rnn_type == 'gru':
            # self.init_hidden = self._gru_init_hidden
            # raise NotImplementedError
            self.encoder = nn.GRU(self.c_emb_dim, self.encoder_hidden_dim, 
                                   self.encoder_layer_num, bidirectional=True, 
                                   batch_first=True)
        elif self.rnn_type == 'coupled_lstm':
            # self.init_hidden = self._coupled_lstm_init_hidden
            # raise NotImplementedError
            self.encoder = CoupledLSTM(self.c_emb_dim, self.encoder_hidden_dim, 
                                   self.encoder_layer_num, bidirectional=True, 
                                   batch_first=True)
        else:
            raise NotImplementedError

        # Compute dims
        if self.pos_sp:
            # -1 for UNK and +1 for the pos
            self.feat_pos_concat_dim = self.f_emb_dim * len(self.vocab_feat)
        else:
            # do not count UNKs
            self.feat_pos_concat_dim = self.f_emb_dim * (len(self.vocab_feat) + len(self.vocab_pos) - 2)
        # times 2 for bi-directional
        self.deocer_inp_dim = self.a_emb_dim + (2 * self.encoder_hidden_dim) + self.feat_pos_concat_dim

        # Decoder
        if self.rnn_type == 'lstm':
            self.decoder = nn.LSTM(self.deocer_inp_dim, self.decoder_hidden_dim, 
                                   self.decoder_layer_num, bidirectional=False)
        elif self.rnn_type == 'gru':
            self.decoder = nn.GRU(self.deocer_inp_dim, self.decoder_hidden_dim, 
                                  self.decoder_layer_num, bidirectional=False)
        elif self.rnn_type == 'coupled_lstm':
            self.decoder = CoupledLSTM(self.deocer_inp_dim, self.decoder_hidden_dim, 
                                       self.decoder_layer_num, bidirectional=False)
        else:
            raise NotImplementedError

        # Classifier
        self.classifier = nn.Linear(self.decoder_hidden_dim, len(self.vocab_act))

    '''
    def _lstm_init_hidden(self, num_layers, num_directions, batch_size, hidden_size):
        h0 = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        c0 = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        return (h0, c0)

    def _gru_init_hidden(self, num_layers, num_directions, batch_size, hidden_size):
        raise NotImplementedError
        pass

    def _coupled_lstm_init_hidden(self, num_layers, num_directions, batch_size, hidden_size):
        raise NotImplementedError
        pass
    '''

    def _loss(self, log_ps, valid_actions, optimal_acts):
        """ @TODO: this is designed for our special case. try to generalize it
        """
        device = log_ps.device
        min_a = valid_actions[0]
        opt_perm_acts = [a - min_a for a in optimal_acts]
        opt_log_ps = torch.index_select(log_ps, -1, torch.tensor(opt_perm_acts, device=device))
        loss = -torch.logsumexp(opt_log_ps, dim=-1)
        return loss

    def _valid_acts(self, curr, seqlen):
        diff = seqlen - curr
        if diff > 1:
            ret = self.vocab_act.unspecial_ids + [self.copy, self.delete]
        elif diff == 1:
            ret = [self.vocab_act.eos] + self.vocab_act.unspecial_ids
        else:
            raise ValueError
        return ret

    def _is_valid_act(self, act, valid_actions):
        """step once
            @TODO: this is designed for our special case. try to generalize it. valid_actions is continuous
        """
        return (act >= valid_actions[0] and act <= valid_actions[-1])
             
    def _predict_one(self, encode_lemma, acts, curr, seqlen, fi, hx):
        """step once
            @TODO: try double feat and MLP
        """
        device = encode_lemma.device

        buff_top_emb = encode_lemma[curr]  # (c_emb_dim,)

        last_act_emb = self.embedding_act(torch.tensor(acts[-1], device=device))  # (a_emb_dim,)

        decoder_input = torch.cat((buff_top_emb, fi, last_act_emb), dim=0)  # (deocer_inp_dim,)

        decoder_input = decoder_input.unsqueeze(0).unsqueeze(0)  # (1, 1, deocer_inp_dim)

        decoder_output, hx = self.decoder(decoder_input, hx)  # (1, 1, decoder_hidden), (1, 1, decoder_hidden) (layers and number of hidden vectors are omitted)

        scores = self.classifier(decoder_output).squeeze()  # (len(vocab_act),)

        valid_actions = self._valid_acts(curr, seqlen)  # (len(valid_actions),)

        log_ps = F.log_softmax(torch.index_select(scores, -1, torch.tensor(valid_actions, device=device)), dim=-1)  # (len(valid_actions),)

        return log_ps, hx, valid_actions

    def transduce(self, lemma, acts):
        curr = 0
        word = []
        for i, act in enumerate(acts):
            if act == self.copy:
                word.append(lemma[curr].item())
                curr += 1
            elif act == self.delete:
                curr += 1
            elif act == self.vocab_act.eos:
                word.append(self.vocab_act.eos)
            else:
                word.append(act)
        return word

    def actions_cost(self, acts, lemma, target, target_len):
        beta = 5
        #pred = self.transduce(lemma, acts)
        #ed = distance(target[:target_len], pred)
        ed = 0
        cost = 0
        for a in acts:
            if a != self.copy:
                cost += 1
        return beta * ed + cost

    def beam_decode(self, encode_lemma, acts, curr, seqlen, fi, hx, beam_width=4):
        best_complete_acts = None  # (acts, log_p)
        beam = [(list(acts), 0.0, curr, None)]
        act_num = len(acts)

        while act_num <= MAX_ACTION_NUM:
            new_beam = []
            for acts, log_p, curr, hx in beam:
                log_ps, hx, valid_actions = self._predict_one(encode_lemma, acts, curr, seqlen, fi, hx)
                top_log_ps, top_perm_acts = log_ps.sort(descending=True)
                for i in range(min(beam_width, len(top_log_ps))):
                    log_pi = log_p + top_log_ps[i]
                    perm_acti = top_perm_acts[i]
                    acti = valid_actions[perm_acti]  # convert pseudo id to real id
                    if best_complete_acts is not None and log_pi < best_complete_acts[1]:  # can not find better
                        continue
                    new_acts = acts + [acti]
                    if acti == self.copy:
                        new_beam.append((new_acts, log_pi, curr + 1, hx))
                    elif acti == self.delete:
                        new_beam.append((new_acts, log_pi, curr + 1, hx))
                    elif acti != self.vocab_act.eos:  # insert
                        new_beam.append((new_acts, log_pi, curr, hx))
                    elif acti == self.vocab_act.eos:
                        if best_complete_acts is None or log_pi > best_complete_acts[1]:  # find better sequence
                            best_complete_acts = (new_acts, log_pi)
                    else:
                        raise ValueError
            if len(new_beam) == 0:
                assert best_complete_acts is not None
                break
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
            act_num += 1

        if best_complete_acts is None:  # no complete action sequence found
            best_complete_acts = (beam[0][0], beam[0][1])
        #print(len(best_complete_acts[0]))
        return best_complete_acts  # (acts, log_p)

    def model_roll_out(self, encode_lemma, lemma_len, fi, hx, target, curr, 
                       buffer_top_i, correct_num, valid_actions, beam_width=4, lemma=None, target_len=None):
        potential_act_costs = []
        if self._is_valid_act(self.copy, valid_actions) and buffer_top_i == target[correct_num]:  # copy
            best_acts, _ = self.beam_decode(encode_lemma, [self.copy], curr + 1, 
                                            lemma_len, fi, hx, beam_width=beam_width)
            potential_act_costs.append((self.copy, self.actions_cost(best_acts, lemma, target, target_len)))
        if self._is_valid_act(self.delete, valid_actions):  # delete
            best_acts, _ = self.beam_decode(encode_lemma, [self.delete], curr + 1, 
                                            lemma_len, fi, hx, beam_width=beam_width)
            potential_act_costs.append((self.delete, self.actions_cost(best_acts, lemma, target, target_len)))
        target_i = target[correct_num].item()
        if self._is_valid_act(target_i, valid_actions):  # insert
            best_acts, _ = self.beam_decode(encode_lemma, [target_i], curr, 
                                            lemma_len, fi, hx, beam_width=beam_width)
            potential_act_costs.append((target_i, self.actions_cost(best_acts, lemma, target, target_len)))
        return potential_act_costs
    
    def _edit_cost_dp(self, lemma, lemma_len, target, target_len):
        dp = np.zeros((lemma_len+1, target_len+1), dtype=np.int64)
        for j in range(1, target_len+1):
            dp[0][j] = j
        for i in range(1, lemma_len+1):
            dp[i][0] = i
            for j in range(1, target_len+1):
                if lemma[i-1] == target[j-1]:
                    dp[i][j] = min(dp[i-1][j-1],    # copy
                                   dp[i-1][j] + 1,  # del
                                   dp[i][j-1] + 1)  # ins
                else:
                    dp[i][j] = min(dp[i-1][j] + 1,  # del
                                   dp[i][j-1] + 1)  # ins
        return dp[-1][-1]

    def expert_roll_out(self, lemma, lemma_len, target, target_len, curr, buffer_top_i, correct_num, valid_actions):
        potential_act_costs = []
        if self._is_valid_act(self.copy, valid_actions) and buffer_top_i == target[correct_num]:  # copy
            copy_cost = self._edit_cost_dp(lemma[curr+1:], lemma_len-curr-1, target[correct_num+1:], target_len-correct_num-1)
            potential_act_costs.append((self.copy, copy_cost))
        if self._is_valid_act(self.delete, valid_actions):  # delete
            del_cost = 1 + self._edit_cost_dp(lemma[curr+1:], lemma_len-curr-1, target[correct_num:], target_len-correct_num)
            potential_act_costs.append((self.delete, del_cost))
        target_i = target[correct_num].item()
        if self._is_valid_act(target_i, valid_actions):  # insert
            ins_cost = 1 + self._edit_cost_dp(lemma[curr:], lemma_len-curr, target[correct_num+1:], target_len-correct_num-1)
            potential_act_costs.append((target_i, ins_cost))
        return potential_act_costs

    def roll_out(self, lemma, encode_lemma, lemma_len, fi, hx, target, 
                 target_len, curr, correct_num, valid_actions, 
                 use_model_roll_out, beam_width=4,):
        costs = [sys.maxsize] * len(self.vocab_act)
        optimal_acts = []
        buffer_top_i = lemma[curr]

        if correct_num == target_len - 1:  # remaining only eos
            if self._is_valid_act(self.delete, valid_actions):  # buffer is not empty
                optimal_acts = [self.delete]
                costs[self.delete] = 0
            elif self._is_valid_act(self.vocab_act.eos, valid_actions):
                optimal_acts = [self.vocab_act.eos]
                costs[self.vocab_act.eos] = 0
            else:
                raise ValueError
        elif correct_num < target_len - 1:
            target_i = target[correct_num]
            if use_model_roll_out:
                potential_act_costs = self.model_roll_out(encode_lemma, lemma_len, fi, hx, target, curr, buffer_top_i, correct_num, valid_actions, beam_width=beam_width, lemma=lemma, target_len=target_len)
            else:
                potential_act_costs = self.expert_roll_out(lemma, lemma_len, target, target_len, curr, buffer_top_i, correct_num, valid_actions)
            opt_cost = min(list(zip(*potential_act_costs))[1])
            for act, cost in potential_act_costs:
                r_cost = cost - opt_cost
                costs[act] = r_cost
                if r_cost == 0:
                    optimal_acts.append(act)
        else:
            raise ValueError
        
        return optimal_acts, costs

    def model_roll_in(self, log_ps, valid_actions):
        agg = 0.0
        r = np.random.rand()
        ps = torch.exp(log_ps)
        for i, act in enumerate(valid_actions):
            agg += ps[i]
            if agg >= r:
                return act
        return valid_actions[-1]

    def expert_roll_in(self, log_ps, valid_actions, optimal_acts):
        """ @TODO: this is designed for our special case. try to generalize it
        """
        min_a = valid_actions[0]
        return optimal_acts[np.argmax([log_ps[a-min_a] for a in optimal_acts])]

    def roll_in(self, log_ps, valid_actions, optimal_acts, use_model_roll_in):
        if use_model_roll_in:
            action = self.model_roll_in(log_ps, valid_actions)
        else:
            action = self.expert_roll_in(log_ps, valid_actions, optimal_acts)
        return action

    def forward(self, lemma, lemma_len, feat, pos, m_lemma, target=None, 
                target_len=None, model_roll_in_p=0.5, model_roll_out_p=0.5, 
                beam_width=1):
        """Transduce lemmas and get loss and predictions

        @NOTE: roll-in method is sampled at each action (local) @TODO: add global
        @NOTE: roll-out method is sampled at each instance (global) @TODO: add local
        
        Args:
            lemma: the padded input form of the words. (batch, seqlen)
            lemma_len: the lengths of the lemmas. (batch,)
            feat: input grammar features. (batch, len(vocab_feat)-1); will refer as (batch, nfeat) later
            pos: input POS tags. (batch, 1) if self.pos_sp else (batch, len(vocab_pos)-1); will refer as (batch, npos) later
            model_roll_in_p: the probability to use model roll in
            model_roll_out_p: the probability to use model roll out
            target: the padded target word form. None means the model will predict. (batch, seqlen)
            target_len: the lengths of the targets. (batch,)

        Returns:
            loss:
            prediction:
            predicted_acts:
        """
        device = lemma.device
        batch_size = lemma.size()[0]

        # Check running type
        test = False if target is not None else True

        vlemma = self.embedding_char(lemma)  # (batch, seqlen, c_emb_dim)

        # Get feature and pos vec
        vpos = self.embedding_pos(pos)  # (batch, npos, f_emb_dim)
        vfeat = self.embedding_feat(feat)  # (batch, nfeat, f_emb_dim)
        f = torch.cat((vpos, vfeat), dim=1).view(batch_size, self.feat_pos_concat_dim)  # (batch, feat_pos_concat_dim)

        # Encode
        # h0 = self.init_hidden(self.encoder_layer_num, 2, batch_size, self.encoder_hidden_dim)
        # h0 will be init to zero by default
        encode_lemmas, _ = self.encoder(vlemma)  # (batch, seqlen, num_directions * hidden_size)

        batch_loss = torch.tensor(0.0, device=device)
        words = []
        action_histories = []

        loss_items = 0
        for i in range(batch_size):
            action_history = [self.copy]
            word = [self.vocab_act.bos]
            act_num = 1
            curr = 1
            correct_num = 1
            use_model_roll_out = (np.random.rand() < model_roll_out_p)

            seqlen = lemma_len[i]
            fi = f[i]
            hx = None

            if not test:  # train
                while act_num < MAX_ACTION_NUM:
                    log_ps, hx, valid_actions = self._predict_one(encode_lemmas[i], 
                                                                action_history, 
                                                                curr, seqlen, fi, 
                                                                hx)

                    optimal_acts, costs = self.roll_out(lemma[i], encode_lemmas[i], 
                                                        lemma_len[i], fi, hx, 
                                                        target[i], target_len[i], 
                                                        curr, correct_num, 
                                                        valid_actions, 
                                                        use_model_roll_out, 
                                                        beam_width=1)

                    use_model_roll_in = (np.random.rand() < model_roll_in_p)

                    act = self.roll_in(log_ps, valid_actions, optimal_acts, use_model_roll_in)

                    # record the newest action
                    action_history.append(act)

                    # transduce character
                    if act == self.copy:
                        word.append(lemma[i][curr].item())
                        curr += 1
                        if word[-1] == target[i][correct_num]:
                            correct_num += 1
                    elif act == self.delete:
                        curr += 1
                    elif act == self.vocab_act.eos:
                        word.append(self.vocab_act.eos)
                        curr += 1
                        if word[-1] == target[i][correct_num]:
                            correct_num += 1
                        break
                    else:
                        word.append(act)
                        if word[-1] == target[i][correct_num]:
                            correct_num += 1

                    step_loss = self._loss(log_ps, valid_actions, optimal_acts)

                    batch_loss = batch_loss + step_loss
                    loss_items += 1

                    act_num += 1

            else:  # test
                action_history, _ = self.beam_decode(encode_lemmas[i], action_history, curr, seqlen, fi, hx, beam_width=beam_width)
                word = self.transduce(m_lemma[i], action_history)

            words.append(word)
            action_histories.append(action_history)

            batch_loss = batch_loss / loss_items

        return batch_loss, words, action_histories
