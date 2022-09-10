from copy import deepcopy
from torch.utils.data import Dataset
from data.datasets.sqlite_db import SQLiteDB
from utils import load_json, highlight_positions
from utils.text_utils import clean_text


class CscDataset(Dataset):
    def __init__(self, fp):
        self.skip_clean = True
        self.skip_wrong_ids = True
        self.data_postfix = fp.split('.')[-1]

        if self.data_postfix in ['json']:
            self.data = load_json(fp)
        elif self.data_postfix in ['txt']:
            self.data = [line.strip() for line in open(fp, 'r')]
        elif self.data_postfix in ['pkl']:
            import pickle
            self.data = pickle.load(open(fp, 'rb'))
        elif self.data_postfix in ['db']:
            self.data = SQLiteDB(db_path=fp, load_now=True)
        else:
            raise ValueError(f"Invalid fp postfix: {self.data_postfix}")

    def __len__(self):
        return self.data.__len__()

    def generate_wrong_ids(self, ot, ct):
        # 这里采用的是字符粒度
        # 我们在使用BERT模型时通常需要重建为 tokenizer 分词后的粒度
        return [_i for _i, (_o, _c) 
                in enumerate(zip(ot, ct)) if _o != _c]

    def show_item(self, tup):
        if isinstance(tup, int):
            tup = self[tup]
        if isinstance(tup, list):
            if len(tup) < 3:
                tup.append(self.generate_wrong_ids(tup[0], tup[1]))
            highlight_positions(
                text=tup[0], positions=tup[2], color='blue')
            highlight_positions(
                text=tup[1], positions=tup[2], color='blue')
        elif isinstance(tup, dict):
            if 'wrong_ids' not in tup:
                tup['wrong_ids'] = self.generate_wrong_ids(
                    tup['original_text'], tup['correct_text'])
            highlight_positions(
                text=tup['original_text'], positions=tup['wrong_ids'], color='blue')
            highlight_positions(
                text=tup['correct_text'], positions=tup['wrong_ids'], color='blue')

    def __getitem__(self, index):
        wr_ids = None

        if self.data_postfix in ['json', 'pkl']:
            ot = self.data[index]['original_text']
            ct = self.data[index]['correct_text']
            wr_ids = self.data[index].get('wrong_ids')
        elif self.data_postfix in ['txt']:
            t_case = self.data[index].split('\t')
            ot = t_case[0].strip()
            if len(t_case) > 1:
                ct = t_case[1].strip()
            else:
                ct = ot
        elif self.data_postfix in ['db']:
            t_case = deepcopy(self.data[index]).split('\t')
            ot, ct = t_case[0].strip(), t_case[1].strip()
        else:
            raise ValueError(f"Invalid fp postfix: {self.data_postfix}")
        
        if not self.skip_clean:
            ot = clean_text(ot)  
            ct = clean_text(ct)  
        if wr_ids is None and not self.skip_wrong_ids:
            wr_ids = self.generate_wrong_ids(ot, ct)
        return ot, ct, wr_ids

