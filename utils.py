""" utility functions"""
import re
import os
from os.path import basename
import subprocess

import gensim
import torch
from torch import nn

from evaluate import eval_rouge
from ConfManager import ConfManager


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3


def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  # word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    return sorted_gpu_info


cm = ConfManager()


def calc_official_rouge(dec_dir, name):
    if name == 'val':
        ref_dir = cm.REF04
    else:
        ref_dir = cm.REF11
    print(f'{name}: ref_dir={ref_dir}')
    dec_pattern = r'(\d+).dec'
    ref_pattern = '#ID#.[A-Z].ref'
    output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
    # print(output)
    for line in output.split('\n'):
        if line.startswith('1 ROUGE-1 Average_F'):
            r1 = float(line.split()[3])
        if line.startswith('1 ROUGE-2 Average_F'):
            r2 = float(line.split()[3])
        if line.startswith('1 ROUGE-L Average_F'):
            rl = float(line.split()[3])
        if line.startswith('1 ROUGE-SU4 Average_F'):
            rsu4 = float(line.split()[3])
    R = {'R-1': r1, 'R-2': r2, 'R-L': rl, 'R-SU4': rsu4}
    print(R, '\n')
    return R


def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    if not logger:
        print("\n" + info + "\n")
    else:
        logger.info("\n" + info + "\n")
