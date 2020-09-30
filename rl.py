""" RL training utilities"""
import math
import pickle
from time import time
from datetime import timedelta
import os
from os.path import join
from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n
from model.ScoreAgent import is_quote
from training import BasicPipeline
from utils import calc_official_rouge


def a2c_validate(agent, abstractor, loader, save_dir, n_epochs, name):
    official_eval = True
    n_epochs -= 1
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    n_sent = []
    sent_idx_l = []
    with torch.no_grad():
        for art_batch, abs_batch, name_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts, dname in zip(art_batch, name_batch):
                # split by ###
                raw_arts_l = [[]]
                raw_arts_filter = []
                for sent in raw_arts:
                    if sent[0] == '###':
                        raw_arts_l.append([])
                    elif is_quote(sent):
                        continue
                    else:
                        raw_arts_l[-1].append(sent)
                        raw_arts_filter.append(sent)
                raw_arts_l = [raw_arts for raw_arts in raw_arts_l if len(raw_arts) > 0]
                indices = agent(raw_arts_l, dname=dname, dataset=name)
                # indices = agent(raw_arts)
                ext_inds += [(len(ext_sents), len(indices) - 1)]
                n_sent.append(len(indices) - 1)
                ext_sents += [raw_arts_filter[idx.item()]
                              for idx in indices if idx.item() < len(raw_arts_filter)]
                sent_idx_l.append([idx.item() for idx in indices if idx.item() < len(raw_arts_filter)])
            # ext or ext+abs
            # all_summs = abstractor(ext_sents)
            all_summs = ext_sents
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j + n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                if official_eval:
                    if not os.path.exists(join(save_dir, 'dec_all', f'{name}-{str(n_epochs)}', 'dec')):
                        os.system(f"mkdir -p {join(save_dir, 'dec_all', f'{name}-{str(n_epochs)}', 'dec')}")
                    fname = join(save_dir, 'dec_all', f'{name}-{str(n_epochs)}', 'dec', f'{i}.dec')
                    with open(fname, 'w') as o:
                        o.write(' '.join(word for word in list(concat(summs))))
                i += 1
    avg_reward /= (i / 100)
    print(f'finished in {timedelta(seconds=int(time() - start))}! unofficial ROUGE-1: {avg_reward:.2f}, '
          f'avg n_sent: {np.mean(n_sent):.2f}')
    metric = {'reward': avg_reward}
    if official_eval:
        fname = join(save_dir, 'dec_all', f'{name}-{str(n_epochs)}', 'sent_idx_l.pkl')
        pickle.dump(sent_idx_l, open(fname, 'wb'))
        official_rouge = calc_official_rouge(join(save_dir, 'dec_all', f'{name}-{str(n_epochs)}', 'dec'), name)
        metric.update(official_rouge)
        metric['reward'] = metric['R-1']  # use official R-1 in saved_model name now
    return metric


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0, n_epochs=1):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    n_sent = []
    diversity_l = []
    art_batch, abs_batch, name_batch = next(loader)
    for raw_arts, dname in zip(art_batch, name_batch):
        # split by ###
        raw_arts_l = [[]]  # each ele is a doc (list of sent)
        raw_arts_filter = []  # each ele is a sent
        for sent in raw_arts:
            if sent[0] == '###':
                raw_arts_l.append([])
            elif is_quote(sent):
                continue
            else:
                raw_arts_l[-1].append(sent)
                raw_arts_filter.append(sent)
        raw_arts_l = [raw_arts for raw_arts in raw_arts_l if len(raw_arts) > 0]
        (inds, ms), bs, diversity = agent(raw_arts_l, dname=dname, dataset='train', n_epochs=n_epochs)
        diversity_l.append(diversity)
        n_sent.append(len(inds) - 1)
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        ext_sents += [raw_arts_filter[idx.item()]
                      for idx in inds if idx.item() < len(raw_arts_filter)]
    # ext or ext+abs
    summaries = ext_sents
    # with torch.no_grad():
    #     summaries = abstractor(ext_sents)
    i = 0
    rewards = []
    avg_reward = 0
    max_len = 10000
    for inds, abss, raw_arts, diversity in zip(indices, abs_batch, art_batch, diversity_l):
        # ROUGE-L each sentence against whole ref / one ref sentence abss[j]
        # 0 for sentences more than len(abss)
        # 10 * ROUGE-1 whole_ext vs. whole_ref for rewarding STOP
        rs = ([reward_fn(summaries[i + j], list(concat(abss)))  # abss[j]
               for j in range(min(len(inds) - 1, len(abss)))]
              + [0 for _ in range(max(0, len(inds) - 1 - len(abss)))]
              + [stop_coeff * stop_reward_fn(
                    list(concat(summaries[i:i + len(inds) - 1]))[:max_len],
                    list(concat(abss)))])

        assert len(rs) == len(inds)
        # avg_reward += np.mean(mmr_scores)
        r1 = stop_reward_fn(list(concat(summaries[i:i + len(inds) - 1]))[:100], list(concat(abss)))
        avg_reward += r1
        # avg_reward += rs[-1] / stop_coeff
        i += len(inds) - 1
        # compute discounted rewards
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
            reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        # losses.append(-p.log_prob(action) * advantage)
        losses.append(-p.log_prob(action) * (advantage / len(indices)))  # divide by T*B
    critic_loss = F.mse_loss(baseline, reward).reshape(1)
    # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.ones(1).to(critic_loss.device)] * (1 + len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward / len(art_batch)
    log_dict['advantage'] = avg_advantage.item() / len(indices)
    log_dict['mse'] = critic_loss.item()
    log_dict['avg_n_sent'] = np.mean(n_sent)
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]

    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1 / 2)
            grad_log['grad_norm' + n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        # grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log

    return f


class EMA:
    def __init__(self, gamma, model, prev_shadow=None):
        super(EMA, self).__init__()
        self.gamma = gamma
        self.model = model
        if prev_shadow is not None:
            self.shadow = prev_shadow
        else:
            self.shadow = {}
            for name, para in model.named_parameters():
                if para.requires_grad:
                    self.shadow[name] = para.clone()

    def update(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = (1.0 - self.gamma) * para + self.gamma * self.shadow[name]

    def swap_parameters(self):
        # swap the shadow parameters and model parameters.
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                para.data += self.shadow[name].data
                self.shadow[name].data = self.shadow[name].data.neg_()
                self.shadow[name].data += para.data
                para.data -= self.shadow[name].data

    def state_dict(self):
        return self.shadow


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher, test_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._test_batcher = test_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

        self.using_ema = None
        if self.using_ema is not None:
            self.ema = EMA(gamma=.1, model=self._net, prev_shadow=None)

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        if self.using_ema is not None and self.using_ema:
            self.ema.swap_parameters()
            self.using_ema = False
        self._n_epoch += 1
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff, self._n_epoch
        )
        if self.using_ema is not None:
            self.ema.update()
        return log_dict

    def validate(self, save_dir, n_epochs, name):
        if self.using_ema is not None and not self.using_ema:
            self.ema.swap_parameters()
            self.using_ema = True
        if name == 'val':
            return a2c_validate(self._net, self._abstractor, self._val_batcher, save_dir, n_epochs, name)
        if name == 'test':
            return a2c_validate(self._net, self._abstractor, self._test_batcher, save_dir, n_epochs, name)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
