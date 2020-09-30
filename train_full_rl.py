""" full training (train rnn-ext + abs + RL) """
import argparse
import json
import pickle as pkl
import os
from datetime import datetime
from os.path import join, exists
from itertools import cycle

from toolz.sandbox.core import unzip
from cytoolz import identity

from ConfManager import ConfManager
from utils import get_gpu_memory_map, print_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sorted_gpu_info = get_gpu_memory_map()
for gpu_id, (mem_left, util) in sorted_gpu_info:
    if mem_left >= 2000:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # args.cuda = f'cuda:{gpu_id}'
        print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
        break
else:
    print(f'no gpu has memory left >=  MB, exiting...')
    exit()

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.data import CnnDmDataset
from data.batcher import tokenize

from model.rl import ActorCritic
from model.extract import PtrExtractSumm

from training import BasicTrainer
from rl import get_grad_fn
from rl import A2CPipeline
from decoding import load_best_ckpt
from decoding import Abstractor, ArticleBatcher
from metric import compute_rouge_l, compute_rouge_n

import random
import numpy as np

seed = 111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_ABS_LEN = 30


class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""

    def __init__(self, split):
        super().__init__(split, os.environ["DATA"])

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents, js_data['id']


def load_ext_net(ext_dir):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    assert ext_meta['net'] == 'ml_rnn_extractor'
    ext_ckpt = load_best_ckpt(ext_dir)
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
    ext = PtrExtractSumm(**ext_args)
    ext.load_state_dict(ext_ckpt)
    return ext, vocab


def configure_net(abs_dir, ext_dir, cuda):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(ext_dir)
    agent = ActorCritic(extractor._sent_enc,
                        extractor._art_enc,
                        extractor._extractor,
                        ArticleBatcher(agent_vocab, cuda))
    if cuda:
        agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, abstractor, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer'] = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size'] = batch_size
    train_params['lr_decay'] = lr_decay
    train_params['gamma'] = gamma
    train_params['reward'] = reward
    train_params['stop_coeff'] = stop_coeff
    train_params['stop_reward'] = stop_reward

    return train_params


def build_batchers(batch_size):
    def coll(batch):
        art_batch, abs_batch, i_batch = unzip(batch)
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents, list(i_batch)

    loader = DataLoader(
        RLDataset('train'), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset('val'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    test_loader = DataLoader(
        RLDataset('test'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader, test_loader


def train(args):
    if not exists(args.path):
        os.makedirs(args.path)

    # make net
    agent, agent_vocab, abstractor, net_args = configure_net(
        args.abs_dir, args.ext_dir, args.cuda)

    # configure training setting
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )
    train_batcher, val_batcher, test_batcher = build_batchers(args.batch)
    reward_fn = compute_rouge_l
    stop_reward_fn = compute_rouge_n(n=1)

    # save abstractor binary
    if args.abs_dir is not None:
        abs_ckpt = {}
        abs_ckpt['state_dict'] = load_best_ckpt(args.abs_dir)
        abs_vocab = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
        abs_dir = join(args.path, 'abstractor')
        os.makedirs(join(abs_dir, 'ckpt'))
        with open(join(abs_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['abstractor'], f, indent=4)
        torch.save(abs_ckpt, join(abs_dir, 'ckpt/ckpt-0-0'))
        with open(join(abs_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(abs_vocab, f)
    # save configuration
    meta = {}
    meta['net'] = 'rnn-ext_abs_rl'
    meta['net_args'] = net_args
    meta['train_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    # for name, param in agent.named_parameters():
    #     print('{} {} {}'.format(name, type(param.data), param.size()))
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    # NB no scheduler
    scheduler = None
    # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
    #                               factor=args.decay, min_lr=0,
    #                               patience=args.lr_p)

    pipeline = A2CPipeline(meta['net'], agent, abstractor,
                           train_batcher, val_batcher, test_batcher,
                           optimizer, grad_fn,
                           reward_fn, args.gamma,
                           stop_reward_fn, args.stop)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score', args=args)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()
    print(f'[{remark}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path', default='saved_model/abs_sent')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model', default='saved_model/ext_sent')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')
    # training options
    parser.add_argument('--reward', action='store', default='rouge-l',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, action='store', default=0,
                        help='weight_decay')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=5,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=10.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=8,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=100,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=30,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    cm = ConfManager()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.data = cm.TRAIN_SC
    remark = f'data{args.data}_{cm.mode}_{cm.remark}'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.path = os.path.join(cm.save_path, current_time + '_' + remark)
    print_config(args)
    os.environ["DATA"] = os.path.join(cm.DATA, args.data)
    train(args)
