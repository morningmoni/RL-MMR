import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

from ConfManager import ConfManager

cm = ConfManager()
sys.path.append(cm.BASE)

from data.data_info import DataInfo
from metric import compute_rouge_l
import pickle
from tqdm import tqdm


def create_sc():
    print('[ScoreAgent.py create_sc()] Data:', cm.TRAIN_SC, cm.VAL_SC, cm.TEST_SC)
    sc = {'train': ScoreAgent(cm.TRAIN_SC, cm.I_MODE, cm.D_MODE), 'val': ScoreAgent(cm.VAL_SC, cm.I_MODE, cm.D_MODE),
          'test': ScoreAgent(cm.TEST_SC, cm.I_MODE, cm.D_MODE)}
    return sc


class ScoreAgent:
    def __init__(self, dataset, i_mode='tfidf', d_mode='tfidf'):
        doc_all = []
        if dataset == '010203' or dataset == '080910':
            dataset_l = [dataset[0:2], dataset[2:4], dataset[4:6], ]
            for d in dataset_l:
                data_info = DataInfo(d)
                for i in range(data_info.n_total):
                    with open(f'{data_info.perdoc_path}/{i}.json') as f:
                        data = json.loads(f.read())
                    doc = ' '.join(data['article'])
                    doc_all.append(doc)
            data_info = DataInfo(dataset)
        else:
            data_info = DataInfo(dataset)
            for i in range(data_info.n_total):
                with open(f'{data_info.perdoc_path}/{i}.json') as f:
                    data = json.loads(f.read())
                doc = ' '.join(data['article'])
                doc_all.append(doc)
        print(f'[ScoreAgent __init__] Dataset={dataset} i_mode={i_mode} d_mode={d_mode}')
        self.tfidf = TfidfVectorizer(max_features=50000)
        self.doc_all_tfidf = self.tfidf.fit_transform(doc_all)
        if data_info.bert_pre is not None:
            self.bert_pre = pickle.load(open(data_info.bert_pre, 'rb'))
        self.i_scores = defaultdict(dict)
        self.dname = None
        self.sim = {}
        if 'rouge' in d_mode:
            self.sim = pickle.load(open(data_info.sim_path, 'rb'))
        elif 'para' in d_mode:
            self.sim = pickle.load(open(data_info.para_pre, 'rb'))
        self.sim = dict(self.sim)
        self.d_scores = defaultdict(dict)
        self.d_mode = d_mode
        self.i_mode = i_mode

    def calc_importance(self, sent_idx, sent_l, avg_each=True):
        if self.i_mode not in self.i_scores[self.dname]:
            if self.i_mode == 'tfidf':
                doc_vec = self.tfidf.transform(sent_l)
                doc_vec_all = self.tfidf.transform([' '.join(sent_l)])
                scores = cosine_similarity(doc_vec, doc_vec_all).squeeze()
            elif self.i_mode == 'bert':
                assert len(sent_l) == len(self.bert_pre[self.dname])
                doc_vec = np.stack(list(self.bert_pre[self.dname].values()))
                doc_vec_all = np.mean(doc_vec, axis=0, keepdims=True)
                scores = cosine_similarity(doc_vec, doc_vec_all).squeeze()
            else:
                raise NotImplementedError
            self.i_scores[self.dname][self.i_mode] = scores
        return self.i_scores[self.dname][self.i_mode][sent_idx]

    def calc_diversity(self, sent_idx, sent_l, cur):
        if len(cur) == 0:
            return 0
        if self.d_mode == 'tfidf' or self.d_mode == 'bert':
            if self.d_mode not in self.d_scores[self.dname]:
                if self.d_mode == 'tfidf':
                    doc_vec = self.tfidf.transform(sent_l)
                else:
                    assert len(sent_l) == len(self.bert_pre[self.dname])
                    doc_vec = np.stack(list(self.bert_pre[self.dname].values()))
                scores = cosine_similarity(doc_vec, doc_vec).squeeze()
                self.d_scores[self.dname][self.d_mode] = scores
            return -max([self.d_scores[self.dname][self.d_mode][sent_idx][i] for i in cur])

        # for i in cur:
        #     if name not in self.sim or (sent_idx, i) not in self.sim[name]:
        #         self.sim[name][sent_idx, i] = compute_rouge_l(sent_l[sent_idx], sent_l[i], mode='p')
        assert self.d_mode in ['rouge', 'para']
        assert len(sent_l) * (len(sent_l) - 1) == len(self.sim[self.dname]), (
            self.dname, len(sent_l), len(self.sim[self.dname]))
        score = -max([self.sim[self.dname][sent_idx, i] for i in cur])
        return score

    def calc_score(self, sent_idx, cur, sent_l, dname, alpha=0.6, beta=.9, return_both=False, min_v=-1,
                   importance_l=None):
        # the beginning of new doc
        if len(cur) == 0:
            self.dname = dname
        # already selected or not in length limit
        if sent_idx in cur or not (8 < len(sent_l[sent_idx].split()) < 55):
            if return_both:
                return min_v, min_v
            return min_v
        if self.i_mode == 'tfidf' or self.i_mode == 'bert':
            I = self.calc_importance(sent_idx, sent_l)
        elif self.i_mode == 'tfidf-self':
            I1 = self.calc_importance(sent_idx, sent_l)
            I2 = importance_l[sent_idx]
            I = beta * I1 + (1 - beta) * I2
        elif self.i_mode == 'tfidf-bert':
            self.i_mode = 'tfidf'
            I1 = self.calc_importance(sent_idx, sent_l)
            self.i_mode = 'bert'
            I2 = self.calc_importance(sent_idx, sent_l)
            self.i_mode = 'tfidf-bert'
            I = beta * I1 + (1 - beta) * I2
        else:
            raise NotImplementedError

        if self.d_mode in ['rouge', 'tfidf', 'bert', 'para']:
            D = self.calc_diversity(sent_idx, sent_l, cur)
        elif self.d_mode == 'tfidf-bert':
            self.d_mode = 'tfidf'
            D1 = self.calc_diversity(sent_idx, sent_l, cur)
            self.d_mode = 'bert'
            D2 = self.calc_diversity(sent_idx, sent_l, cur)
            self.d_mode = 'tfidf-bert'
            D = beta * D1 + (1 - beta) * D2
        elif self.d_mode == 'tfidf-para':
            self.d_mode = 'tfidf'
            D1 = self.calc_diversity(sent_idx, sent_l, cur)
            self.d_mode = 'para'
            D2 = self.calc_diversity(sent_idx, sent_l, cur)
            self.d_mode = 'tfidf-para'
            D = beta * D1 + (1 - beta) * D2
        else:
            raise NotImplementedError

        if return_both:
            return I, D
        return alpha * I + (1 - alpha) * D


def is_quote(tokens, out=False, keep_hash=False):
    contains_quotation_marks = "''" in tokens and len(tokens) > 0 and tokens[0] == "``"
    doesnt_end_with_period = len(tokens) > 0 and tokens[-1] != "."
    # contains_says = "says" in tokens or "said" in tokens
    decision = contains_quotation_marks or doesnt_end_with_period
    if keep_hash and tokens[0] == '###':
        decision = False
    if out and decision:
        print("Skipping quote: ", ' '.join(tokens))
    return decision


def process_one(doc_id, dataset='080910'):
    """
    :param doc_id: an id in [0, n_sets), find n_sets in data_info.py
    :param dataset: can be '010203', '04', '11', '080910'
    :return:
    """
    sim = defaultdict(dict)
    ext = []
    data_info = DataInfo(dataset)
    path = f"{data_info.doc_path}/{doc_id}.json"
    with open(path) as f:
        data = json.loads(f.read())
    name = data['id']
    for ct, sent in enumerate(data['article']):
        if sent[0] == '###':
            continue
        if not is_quote(sent.split()):
            ext.append(sent)
    for i in tqdm(range(len(ext))):
        for j in range(i + 1, len(ext)):
            sim[name][j, i] = compute_rouge_l(ext[j], ext[i], mode='p')
            sim[name][i, j] = compute_rouge_l(ext[i], ext[j], mode='p')
    pickle.dump(sim, open(f'sim_{dataset}_new/sim{doc_id}_{dataset}.pkl', 'wb'))


def combine(dataset='04'):
    data_info = DataInfo(dataset)
    sim = {}
    for doc_id in range(data_info.n_sets):
        data = pickle.load(open(f'sim_{dataset}_new/sim{doc_id}_{dataset}.pkl', 'rb'))
        sim.update(data)
    pickle.dump(sim, open(f'sim_{dataset}_new/sim_{dataset}.pkl', 'wb'))


if __name__ == '__main__':
    pool = Pool(20)
    pool.map(process_one, list(range(138)))
    combine('080910')

    # find max sentence length in a dataset
    # MAX_LEN = 0
    # data_info = DataInfo('03')
    # for i in range(data_info.n_total):
    #     with open(f'{data_info.perdoc_path}/{i}.json') as f:
    #         data = json.loads(f.read())
    #     for sent in data['article']:
    #         MAX_LEN = max(MAX_LEN, len(sent.split()))
    # print(MAX_LEN)

    # avg n_sent in perdoc
    # data_info = DataInfo('04')
    # lens = []
    # for i in range(data_info.n_total):
    #     with open(f'{data_info.perdoc_path}/{i}.json') as f:
    #         data = json.loads(f.read())
    #     lens.append(len(data['article']))
    # import pandas as pd
    # print(pd.Series(lens).describe())

    # DATA = '11'
    # # avg n_sent in doc set
    # data_info = DataInfo(DATA)
    # lens = []
    # for i in range(data_info.n_sets):
    #     with open(f'{data_info.doc_path}/{i}.json') as f:
    #         data = json.loads(f.read())
    #     lens.append(len([i for i in data['article'] if i != '###']))
    # import pandas as pd
    #
    # print(pd.Series(lens).describe())
    #
    # # avg words in doc set
    # data_info = DataInfo(DATA)
    # lens = []
    # for i in range(data_info.n_sets):
    #     with open(f'{data_info.doc_path}/{i}.json') as f:
    #         data = json.loads(f.read())
    #     lens.append(sum([len(sent.split()) for sent in data['article'] if sent != '###']))
    # import pandas as pd
    #
    # print(pd.Series(lens).describe())
