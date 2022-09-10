import time
from data.datasets.example import *
from utils.text_utils import clean_text


class DataCollatorForCsc:
    def __init__(self, tokenizer, cfg=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        self.debug = False
        self.skip_clean = True
        self.timer = []

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        self.timer.append((information, time.time()))

    def show_timer(self):
        phase, start_t = self.timer[0]
        print(phase, time.strftime("%H:%M:%S", time.gmtime(start_t)))
        for phase, t in self.timer[1:]:
            print(phase, '+', t - start_t)

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(
            texts, padding=True, return_tensors='pt')
        # encoded_texts.to(self._device)
        return encoded_texts

    def generate_det_labels(self, original_token_ids, correct_token_ids):
        det_labels = original_token_ids != correct_token_ids
        return det_labels.long()  # .squeeze(-1)

    def generate_model_inputs(self, ori_texts, cor_texts):
        det_labels = self.generate_det_labels(
            self.get_encoded_texts(ori_texts),
            self.get_encoded_texts(cor_texts),
        )
        return ori_texts, cor_texts, det_labels

    def __call__(self, data, debug=False):
        debug = debug or self.debug
        if debug:
            self.record_time("generate starts")

        # ignore original det_labels, and then re-generate one with any tokenizer.
        ori_texts, cor_texts, _ = zip(*data)
        if self.skip_clean:
            ori_texts = [t.strip()[:510] for t in ori_texts if t.strip()]
            cor_texts = [t.strip()[:510] for t in cor_texts if t.strip()]
        else:
            ori_texts = [clean_text(t.strip())[:510] for t in ori_texts if t.strip()]
            cor_texts = [clean_text(t.strip())[:510] for t in cor_texts if t.strip()]
        if debug:
            self.record_time("clean texts")

        drop_no_aligned_pairs = []
        for idx in range(len(ori_texts)):
            if len(ori_texts[idx]) != len(cor_texts[idx]):
                drop_no_aligned_pairs.append(idx)
            elif ori_texts[idx] == "" or cor_texts[idx] == "":
                drop_no_aligned_pairs.append(idx)
        if len(drop_no_aligned_pairs) > 0:
            ori_texts = [t for _i, t in enumerate(ori_texts) if _i not in drop_no_aligned_pairs]
            cor_texts = [t for _i, t in enumerate(cor_texts) if _i not in drop_no_aligned_pairs]
        
        if debug:
            self.record_time("drop no-aligned pairs")

        model_inputs = self.generate_model_inputs(ori_texts, cor_texts)
        if debug:
            self.record_time("generate model inputs")
        return model_inputs


def test_sample_augment(ddc):
    print(ddc.change_words('路遥知马力'))
    data = ([('Overall路遥知马力，日久见人心012345！', 'Overall路遥知马力，日久现人心012345！', [9])])
    item = ddc.sample_augment_single(data)
    print(item)
    for each in ddc(data):
        print(each)


def test_clean_tokens_from_text(ddc):
    # test_clean_tokens_from_text(ddc)
    text = '碪碪她疲惫不碪的halloweenbadapple'
    text = "首先德国laurèl破产原因是其资本结极和制度设计出现问题，幵非品牌自身缺陷；其次国内市场独立二德国市场，公司有独立的设计和销售团队运作该品牌，未来前景依然看好。"
    tokens = ['[CLS]'] + ddc.tokenizer.tokenize(text) + ['[SEP]', '[PAD]']
    print(tokens)
    results = ddc.get_clean_tokens_from_text(tokens=tokens, text=text)
    print(results)
    print(len(tokens), len(results))


if __name__ == "__main__":
    from tqdm import tqdm
    from config import cfg
    from utils import get_abs_path
    config_file='csc/train_cdmac_aug.yml'
    cfg.merge_from_file(get_abs_path('configs', config_file))

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    from data.loaders.collator import *
    ddc = DataCollatorForCsc(
        tokenizer=tokenizer, need_pinyin=True, cfg=cfg)
    ddc.debug = True

    data = ([('购买日之前持有的股权投资，采用金融工具确认和计量准则进行会计处理的，将该股权投资的公允价值加上新增投资成本之和，作为改按成本法核算的初始投资成本，原持有股权的公允价值与账面价值的差额与原计入其他综合收益的累计公允价值变动全部铸入改按成本法核算的当期投资损益。Overall路遥知马力，日久涧人心01234≥5！', 
              '购买日之前持有的股权投资，采用金融工具确认和计量准则进行会计处理的，将该股权投资的公允价值加上新增投资成本之和，作为改按成本法核算的初始投资成本，原持有股权的公允价值与账面价值的差额与原计入其他综合收益的累计公允价值变动全部转入改按成本法核算的当期投资损益。Overall路遥知马力，日久现人心01234≥5！', 
              [138])])
    for _ in tqdm(range(10)):
        ddc(data, debug=True)
    ddc.show_timer()
    
