import random

import torch
from torch.nn.functional import cosine_similarity
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

from ConfManager import ConfManager
from model.mlp import MLP, MLP2
from model.ScoreAgent import create_sc
from .features import calc_feat
from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet

INI = 1e-2


class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""

    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)  # W_p1 h_j
        hop_feat = torch.mm(attn_mem, self._hop_wm)  # W_g1 h_j
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]  # z_t
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                 self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            if self.training:
                prob = F.softmax(score, dim=-1)
                out = torch.distributions.Categorical(prob)
            else:
                for o in outputs:
                    score[0, o[0, 0].item()][0] = -1e18
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            lstm_in = attn_mem[out[0, 0].item()].unsqueeze(0)
            lstm_states = (h, c)
        return outputs

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(torch.tanh(sum_), v.unsqueeze(1)).t()

        return score

    @staticmethod
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(
            PtrExtractorRL.attention_score(attention, query, v, w), dim=-1)
        output = torch.mm(score, attention)

        return output


class PtrExtractorRLStop(PtrExtractorRL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)
        self._stop = nn.Parameter(
            torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI)
        self.sc = create_sc()
        cm = ConfManager()
        self.K = cm.K
        self.beta = cm.beta
        self.mode = cm.mode
        self.use_feat = cm.use_feat
        print(f'K={self.K}, beta={self.beta}, mode={self.mode}, use_feature={self.use_feat}')

        if self.mode == 'alpha':
            self.mlp = MLP2()
        else:
            in_dim = 12 if self.use_feat else 1
            to_score = True if self.mode == 'attention' else True
            self.mlp = MLP(in_dim=in_dim, to_score=to_score)

    def forward(self, attn_mem, n_ext=None, sent_l=None, dname=None, dataset='train', n_epochs=1, enc_sent=None):
        """atten_mem: Tensor of size [num_sents, input_dim]"""

        def filter_by_cond(score):
            # max n_sent
            if len(outputs) == 6:
                score[0, max_step] = 1e18
            # -l 100
            elif sum([len(sent_l[o.item()].split()) for o in outputs]) >= 100:
                score[0, max_step] = 1e18
            # min n_sent
            elif len(outputs) <= 2:
                score[0, max_step] = -1e18
            for o in outputs:
                score[0, o.item()] = -1e18
            return score

        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        max_step = attn_mem.size(0)
        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))

        if self.use_feat:
            feat = calc_feat(sent_l)
            feat = torch.Tensor(feat).to(hop_feat.device)

        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]  # 1 x 256
            cur = [o.item() for o in outputs]
            if self.mode == 'soft-attn':
                mmr_scores = [self.sc[dataset].calc_score(sent_idx, cur, sent_l, dname) for sent_idx in
                              range(len(sent_l))]
                idx_0 = torch.from_numpy(np.argsort(mmr_scores)[:-self.K])
                mmr_scores = torch.Tensor(mmr_scores).to(attn_mem.device).view(-1, 1)
                if self.use_feat:
                    mmr_scores = torch.cat([mmr_scores, feat], dim=1)
                mmr_scores = self.mlp(mmr_scores).view(-1)
                mmr_scores = torch.cat([mmr_scores, torch.Tensor([0]).to(attn_mem.device)]).view(1, -1)

                mmr_scores = F.softmax(mmr_scores * 2, dim=-1)

                attn_mem_mmr = attn_mem * mmr_scores.t()
                attn_feat = torch.mm(attn_mem_mmr, self._attn_wm)
                hop_feat = torch.mm(attn_mem_mmr, self._hop_wm)

            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                 self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)

            assert len(sent_l) == score.shape[1] - 1

            # extract from top-k sentences sorted by MMR
            if self.mode == 'hard-cut':
                mmr_scores = [self.sc[dataset].calc_score(sent_idx, cur, sent_l, dname) for
                              sent_idx in range(len(sent_l))]
                idx_0 = torch.from_numpy(np.argsort(mmr_scores)[:-self.K])
                score[0, idx_0] = -1e18
            # weighted combine
            elif self.mode == 'soft-comb':
                mmr_scores = [self.sc[dataset].calc_score(sent_idx, cur, sent_l, dname) for sent_idx in
                              range(len(sent_l))]
                mmr_scores = torch.Tensor(mmr_scores).to(score.device).view(-1, 1)
                if self.use_feat:
                    mmr_scores = torch.cat([mmr_scores, feat], dim=1)
                mmr_scores = self.mlp(mmr_scores).view(-1)
                mmr_scores = torch.cat([mmr_scores, torch.Tensor([0]).to(score.device)]).view(1, -1)
                score = self.beta * score + (1 - self.beta) * mmr_scores
            # top-k then combine
            elif self.mode == 'hard-comb':
                mmr_scores = [self.sc[dataset].calc_score(sent_idx, cur, sent_l, dname) for sent_idx in
                              range(len(sent_l))]
                idx_0 = torch.from_numpy(np.argsort(mmr_scores)[:-self.K])
                score[0, idx_0] = -1e18
                mmr_scores = torch.Tensor(mmr_scores).to(score.device).view(-1, 1)
                if self.use_feat:
                    mmr_scores = torch.cat([mmr_scores, feat], dim=1)
                mmr_scores = self.mlp(mmr_scores).view(-1)
                mmr_scores = torch.cat([mmr_scores, torch.Tensor([0]).to(score.device)]).view(1, -1)
                score = self.beta * score + (1 - self.beta) * mmr_scores

            score = filter_by_cond(score)

            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            if out.item() == max_step:
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        if dists:
            # return distributions only when not empty (training)
            return outputs, dists
        else:
            return outputs


class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""

    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(torch.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""

    def __init__(self, sent_encoder, art_encoder,
                 extractor, art_batcher):
        super().__init__()
        self._sent_enc = sent_encoder
        self._art_enc = art_encoder
        self._ext = PtrExtractorRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher

    # def forward(self, raw_article_sents, n_abs=None):
    #     article_sent = self._batcher(raw_article_sents)
    #     enc_sent = self._sent_enc(article_sent).unsqueeze(0)
    #     enc_art = self._art_enc(enc_sent).squeeze(0)
    #     if n_abs is not None and not self.training:
    #         n_abs = min(len(raw_article_sents), n_abs)
    #     if n_abs is None:
    #         outputs = self._ext(enc_art)
    #     else:
    #         outputs = self._ext(enc_art, n_abs)
    #     if self.training:
    #         if n_abs is None:
    #             n_abs = len(outputs[0])
    #         scores = self._scr(enc_art, n_abs)
    #         return outputs, scores
    #     else:
    #         return outputs

    def forward(self, raw_article_sents_l, n_abs=None, dname=None, calc_r=False, dataset='04', n_epochs=1):
        # encode doc set
        enc_art_l = []
        enc_sent_l = []
        for raw_article_sents in raw_article_sents_l:
            article_sent = self._batcher(raw_article_sents)
            enc_sent = self._sent_enc(article_sent)
            enc_sent_l.append(enc_sent)
            enc_art = self._art_enc(enc_sent.unsqueeze(0)).squeeze(0)
            enc_art_l.append(enc_art)
        sent_l = [' '.join(sent) for raw_article_sents in raw_article_sents_l for sent in raw_article_sents]
        enc_art = torch.cat(enc_art_l, dim=0)
        enc_sent = torch.cat(enc_sent_l, dim=0)
        if n_abs is None:
            outputs = self._ext(enc_art, sent_l=sent_l, dname=dname, dataset=dataset, n_epochs=n_epochs,
                                enc_sent=enc_sent)
        else:
            outputs = self._ext(enc_art, n_abs)
        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_art, n_abs)
            if calc_r:  # intrinsic diversity measure (not working well
                enc_ext = torch.cat([enc_art[o.item()].view(1, -1) for o in outputs[0][:-1]], dim=0)
                enc_ext_norm = enc_ext / enc_ext.norm(dim=1)[:, None]
                diversity = 1 - torch.mm(enc_ext_norm, enc_ext_norm.transpose(0, 1))
                diversity = diversity.mean()
                return outputs, scores, diversity.item()
            return outputs, scores, 0
        else:
            return outputs
