from pathlib import Path
from ConfManager import ConfManager


class DataInfo:
    def __init__(self, dataset):
        cm = ConfManager()
        BASEPATH = Path(cm.DATA)
        SIMPATH = Path(cm.SIM)

        NEWSIM = '_new'
        # print(f'NEWSIM={NEWSIM == "_new"}')
        assert NEWSIM in ['', '_new']
        base_path = doc_path = sim = n_total = bert_pre = para_pre = None
        if dataset == '04':
            base_path = f"{BASEPATH}/03x"
            doc_path = f"{BASEPATH}/03x/val"
            sim = f'{SIMPATH}/sim_04{NEWSIM}/sim_04.pkl'
            bert_pre = SIMPATH / "emb_04.pkl"
            para_pre = SIMPATH / "sim_04_para.pkl"
            n_sets = 50  # num of samples
            n_total = 500  # num of samples x docs per sample
        elif dataset == '11':
            base_path = f"{BASEPATH}/080910"
            doc_path = f"{BASEPATH}/080910/test"
            sim = f'{SIMPATH}/sim_11{NEWSIM}/sim_11.pkl'
            bert_pre = SIMPATH / "emb_11.pkl"
            # para_pre =  # unavailable
            n_sets = 44
            n_total = 440
        elif dataset == '080910':
            base_path = f"{BASEPATH}/080910"
            doc_path = f"{BASEPATH}/080910/train"
            sim = f'{SIMPATH}/sim_080910{NEWSIM}/sim_080910.pkl'
            n_sets = 138  # 48 + 44 + 46
        elif dataset == '03':
            doc_path = f"{BASEPATH}/03/train"
            sim = f'{SIMPATH}/sim_010203{NEWSIM}/sim_010203.pkl'
            para_pre = SIMPATH / "sim_010203_para.pkl"
            n_sets = 30
            n_total = 298
        else:
            raise NotImplementedError
        self.dataset = dataset
        self.base_path = base_path
        self.doc_path = doc_path
        self.perdoc_path = f"{BASEPATH}/perdoc/perdoc{dataset[:2]}"
        self.sim_path = sim
        self.n_sets = n_sets
        self.n_total = n_total
        self.bert_pre = bert_pre
        self.para_pre = para_pre
