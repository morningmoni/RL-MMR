'''
Parameters in addition to argparse
'''


class ConfManager:
    def __init__(self):
        # paths
        self.BASE = './fast_abs_rl'
        self.DATA = './data/DUC-json/'
        self.SIM = './model'
        self.REF04 = "./data/DUC-json/03x/ref"
        self.REF11 = "./data/DUC-json/080910/ref"
        self.ROUGE = "./pyrouge/tools/ROUGE-1.5.5/"
        self.METEOR = "./meteor-1.5/meteor-1.5.jar"

        # model/rl.py
        self.K = 20  # top-K
        self.beta = .1  # score = self.beta * score + (1 - self.beta) * mmr_scores
        self.mode = 'soft-attn'  # how to combine RL and MMR
        assert self.mode in ['hard-cut', 'hard-comb', 'soft-comb', 'soft-attn']
        self.use_feat = False  # use raw features like sentence position

        # ScoreAgent
        self.TRAIN_SC = '080910'
        assert self.TRAIN_SC in ['03', '080910']
        if self.TEST_SC == '080910':
            self.VAL_SC = '04'
            self.TEST_SC = '11'
        else:
            self.VAL_SC = '11'
            self.TEST_SC = '04'
        self.I_MODE = 'tfidf'
        self.D_MODE = 'tfidf'

        self.remark = 'run1'
        self.save_path = 'saved_model/2020/11'
